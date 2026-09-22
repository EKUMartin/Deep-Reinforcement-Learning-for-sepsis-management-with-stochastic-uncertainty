from pathlib import Path
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Preprocessing.preprocessing import (
    target_cohort,
    states_preprocessor,
    action_preprocessor,
    sofa,
    survival_labeler
)
from Pretraining.encoder import SepsisEncoder
from Pretraining.hmm import SepsisHMM
from Pretraining.sampling import Sample
from policy.policy import LowLevelQNetwork, HighLevelPolicy
from db_conn import db


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target, source, tau):
    for target_param, source_param in zip(
        target.parameters(),
        source.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data
            + (1.0 - tau) * target_param.data
        )


def get_latent(encoder_module, x):
    mu, _ = encoder_module.encoder(x)
    return mu


def get_uncertainty(encoder_module, z):
    t = torch.zeros_like(z[:, :1])

    ty = torch.cat(
        [t, z],
        dim=1
    )

    g_val = (
        encoder_module.sde.g_net(ty)
        + 1e-3
    )

    g_norm = torch.max(
        g_val,
        dim=1
    ).values

    return g_norm


def scale_uncertainty(
    g_norm,
    g_mean,
    g_std
):
    g_standardized = (
        g_norm - g_mean
    ) / (g_std + 1e-6)

    return torch.sigmoid(
        g_standardized
    )


def make_rewards(
    sofa_curr,
    sofa_next,
    survival,
    is_last,
    lambda_pt,
    terminal_reward_value=5.0
):
    reward_step = (
        sofa_curr - sofa_next
    )

    reward_step = torch.clamp(
        reward_step,
        min=-4.0,
        max=4.0
    ) / 4.0

    terminal_reward = torch.where(
        survival == 0,
        torch.full_like(
            reward_step,
            terminal_reward_value
        ),
        torch.full_like(
            reward_step,
            -terminal_reward_value
        )
    )

    reward_q = (
        reward_step
        + is_last.float()
        * terminal_reward
    )

    reward_p = torch.where(
        reward_q >= 0,
        reward_q,
        lambda_pt * reward_q
    )

    return reward_q, reward_p


def cql_loss(q_all, q_data):
    return (
        torch.logsumexp(
            q_all,
            dim=1
        ).mean()
        - q_data.mean()
    )


def make_rl_dataset(
    df,
    features_col,
    next_features_cols,
    scaler
):
    x_curr = scaler.transform(
        df[features_col].values
    )

    x_next = scaler.transform(
        df[next_features_cols].values
    )

    return TensorDataset(
        torch.FloatTensor(
            x_curr
        ),
        torch.FloatTensor(
            x_next
        ),
        torch.LongTensor(
            df['action_to_next'].values
        ),
        torch.FloatTensor(
            df['sofa_score'].values
        ),
        torch.FloatTensor(
            df['sofa_score_next'].values
        ),
        torch.LongTensor(
            df['survival'].values
        ),
        torch.BoolTensor(
            df['is_last'].values
        )
    )


def compute_uncertainty_stats(
    loader,
    encoder_module,
    device
):
    values = []

    encoder_module.encoder.eval()
    encoder_module.sde.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)

            z = get_latent(
                encoder_module,
                x
            )

            g = get_uncertainty(
                encoder_module,
                z
            )

            values.append(
                g.detach().cpu()
            )

    values = torch.cat(
        values,
        dim=0
    )

    mean = values.mean().item()

    std = values.std(
        unbiased=False
    ).item()

    return mean, max(std, 1e-6)


def evaluate(
    loader,
    q_net,
    p_net,
    meta_net,
    target_q,
    target_p,
    encoder_module,
    device,
    gamma,
    lambda_pt,
    cql_weight,
    g_mean,
    g_std
):
    q_net.eval()
    p_net.eval()
    meta_net.eval()
    target_q.eval()
    target_p.eval()

    total_q_loss = 0.0
    total_p_loss = 0.0
    total_meta_loss = 0.0

    total_samples = 0
    hierarchy_correct = 0
    q_correct = 0
    p_correct = 0
    prospect_count = 0

    num_batches = 0

    with torch.no_grad():
        for (
            batch_X,
            batch_X_next,
            batch_act,
            batch_sofa,
            batch_sofa_next,
            batch_surv,
            batch_last
        ) in loader:

            batch_X = batch_X.to(device)
            batch_X_next = batch_X_next.to(device)

            batch_act = batch_act.to(device)

            batch_sofa = batch_sofa.to(device)
            batch_sofa_next = batch_sofa_next.to(device)

            batch_surv = batch_surv.to(device)
            batch_last = batch_last.to(device)

            z_curr = get_latent(
                encoder_module,
                batch_X
            )

            z_next = get_latent(
                encoder_module,
                batch_X_next
            )

            g_norm = get_uncertainty(
                encoder_module,
                z_curr
            )

            g_scaled = scale_uncertainty(
                g_norm,
                g_mean,
                g_std
            )

            reward_q, reward_p = make_rewards(
                batch_sofa,
                batch_sofa_next,
                batch_surv,
                batch_last,
                lambda_pt
            )

            q_all = q_net(
                z_curr
            )

            q_data = q_all.gather(
                1,
                batch_act.unsqueeze(1)
            ).squeeze(1)

            next_action_q = (
                q_net(z_next)
                .argmax(
                    dim=1,
                    keepdim=True
                )
            )

            next_q = (
                target_q(z_next)
                .gather(
                    1,
                    next_action_q
                )
                .squeeze(1)
            )

            q_target = (
                reward_q
                + gamma
                * next_q
                * (~batch_last).float()
            )

            td_q = F.smooth_l1_loss(
                q_data,
                q_target
            )

            cq_q = cql_loss(
                q_all,
                q_data
            )

            loss_q = (
                td_q
                + cql_weight
                * cq_q
            )

            p_all = p_net(
                z_curr
            )

            p_data = p_all.gather(
                1,
                batch_act.unsqueeze(1)
            ).squeeze(1)

            next_action_p = (
                p_net(z_next)
                .argmax(
                    dim=1,
                    keepdim=True
                )
            )

            next_p = (
                target_p(z_next)
                .gather(
                    1,
                    next_action_p
                )
                .squeeze(1)
            )

            p_target = (
                reward_p
                + gamma
                * next_p
                * (~batch_last).float()
            )

            td_p = F.smooth_l1_loss(
                p_data,
                p_target
            )

            cq_p = cql_loss(
                p_all,
                p_data
            )

            loss_p = (
                td_p
                + cql_weight
                * cq_p
            )

            probs = meta_net(
                z_curr,
                g_scaled
            )

            (
                loss_meta,
                _,
                _,
                _
            ) = meta_net.compute_loss(
                probs,
                q_all,
                p_all,
                g_scaled,
                alpha=0.25,
                beta=0.01
            )

            meta_choice = probs.argmax(
                dim=1
            )

            q_action = q_all.argmax(
                dim=1
            )

            p_action = p_all.argmax(
                dim=1
            )

            final_action = torch.where(
                meta_choice == 0,
                q_action,
                p_action
            )

            batch_size = batch_act.size(0)

            total_samples += batch_size

            hierarchy_correct += (
                final_action
                == batch_act
            ).sum().item()

            q_correct += (
                q_action
                == batch_act
            ).sum().item()

            p_correct += (
                p_action
                == batch_act
            ).sum().item()

            prospect_count += (
                meta_choice
                == 1
            ).sum().item()

            total_q_loss += loss_q.item()
            total_p_loss += loss_p.item()
            total_meta_loss += loss_meta.item()

            num_batches += 1

    return {
        'q_loss':
            total_q_loss / num_batches,

        'p_loss':
            total_p_loss / num_batches,

        'meta_loss':
            total_meta_loss / num_batches,

        'total_loss':
            (
                total_q_loss
                + total_p_loss
                + total_meta_loss
            ) / num_batches,

        'hierarchy_agreement':
            hierarchy_correct
            / total_samples,

        'q_agreement':
            q_correct
            / total_samples,

        'p_agreement':
            p_correct
            / total_samples,

        'prospect_ratio':
            prospect_count
            / total_samples
    }


if __name__ == "__main__":

    set_seed(42)

    interval = 4
    rl_epochs = 50
    gamma = 0.99
    lambda_pt = 2.25
    tau = 0.005
    cql_weight = 0.5

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print("Device:", device)

    lab_join = """
        JOIN mimic.icustays i
        ON l.hadm_id = i.hadm_id
    """

    my_required_items = [
        {
            'item_id': 220045,
            'item_name': 'BPM_inv'
        },
        {
            'item_id': 225309,
            'item_name': 'BPM_noninv'
        },
        {
            'item_id': 220546,
            'item_name': 'WBC'
        },
        {
            'item_id': 220179,
            'item_name': 'NIBPs'
        },
        {
            'item_id': 220645,
            'item_name': 'Sodium'
        },
        {
            'item_id': 220621,
            'item_name': 'Glucose'
        },
        {
            'item_id': 220602,
            'item_name': 'Chloride'
        },
        {
            'item_id': 220210,
            'item_name': 'RR'
        },
        {
            'item_id': 224685,
            'item_name': 'Tidal_Volume'
        },
        {
            'item_id': 220224,
            'item_name': 'PaO2'
        },
        {
            'item_id': 223835,
            'item_name': 'FiO2'
        },
        {
            'item_id': 227457,
            'item_name': 'platelets_valuenum'
        },
        {
            'item_id': 227467,
            'item_name': 'zinr'
        },
        {
            'item_id': 220228,
            'item_name': 'HGB'
        },
        {
            'item_id': 227466,
            'item_name': 'PTT'
        },
        {
            'item_id': 220545,
            'item_name': 'Hematocrit'
        },
        {
            'item_id': 220615,
            'item_name': 'creatinine_valuenum'
        },
        {
            'item_id': 225624,
            'item_name': 'BUN'
        },
        {
            'item_id': 227443,
            'item_name': 'bicarbonate'
        },
        {
            'item_id': 224828,
            'item_name': 'Base_Excess_chart'
        },
        {
            'item_id': 225668,
            'item_name': 'Lactate_chart'
        },
        {
            'item_id': 225690,
            'item_name': 'tb_valuenum'
        },
        {
            'item_id': 220587,
            'item_name': 'SGOT'
        },
        {
            'item_id': 227442,
            'item_name': 'Potassium_chart'
        },
        {
            'item_id': 225667,
            'item_name': 'Ionized_Calcium'
        },
        {
            'item_id': 223830,
            'item_name': 'PH'
        },
        {
            'item_id': 225625,
            'item_name': 'Calcium_non_ionized'
        },
        {
            'item_id': 51486,
            'item_name': 'Lab_WBC',
            'table_name': 'mimic_hosp.labevents l',
            'time_col': 'l.charttime',
            'value_col': 'l.valuenum',
            'stay_id_col': 'i.stay_id',
            'join_clause': lab_join,
            'resample_method': 'mean'
        },
        {
            'item_id': 51221,
            'item_name': 'Lab_Hematocrit',
            'table_name': 'mimic_hosp.labevents l',
            'time_col': 'l.charttime',
            'value_col': 'l.valuenum',
            'stay_id_col': 'i.stay_id',
            'join_clause': lab_join,
            'resample_method': 'mean'
        },
        {
            'item_id': 50802,
            'item_name': 'ABE',
            'table_name': 'mimic_hosp.labevents l',
            'time_col': 'l.charttime',
            'value_col': 'l.valuenum',
            'stay_id_col': 'i.stay_id',
            'join_clause': lab_join,
            'resample_method': 'mean'
        },
        {
            'item_id': 50813,
            'item_name': 'Lactate',
            'table_name': 'mimic_hosp.labevents l',
            'time_col': 'l.charttime',
            'value_col': 'l.valuenum',
            'stay_id_col': 'i.stay_id',
            'join_clause': lab_join,
            'resample_method': 'max'
        },
        {
            'item_id': 50971,
            'item_name': 'Potassium',
            'table_name': 'mimic_hosp.labevents l',
            'time_col': 'l.charttime',
            'value_col': 'l.valuenum',
            'stay_id_col': 'i.stay_id',
            'join_clause': lab_join,
            'resample_method': 'mean'
        }
    ]

    initial_values = {
        'NIBPs': 100,
        'NIBPd': 70,
        'heart_rate': 60,
        'SpO2': 90,
        'Temperature': 36,
        'Potassium': 3.5,
        'Glucose': 144,
        'Magnesium': 2.0,
        'SGOT': 5,
        'platelets_valuenum': 5,
        'zinr': 0.8,
        'P': 80,
        'ALT': 7,
        'Sodium': 135,
        'BUN': 7,
        'Calcium': 1.1,
        'tb_valuenum': 0.1,
        'PTT': 35,
        'PH': 7.35,
        'bicarbonate': 22,
        'RR': 12,
        'HGB': 7.0,
        'Chloride': 96,
        'creatinine_valuenum': 0.6,
        'PaCO2': 35,
        'WBC': 4500,
        'PT': 11,
        'pf_ratio': 400,
        'AL': 0.5,
        'F': 21
    }

    zero_fill_cols = [
        'gcs_score',
        'ABE',
        'total_sofa_score',
        'vasopressor_eq',
        'SIRS',
        'shock_index'
    ]

    with_stay = """
        WITH ranked_stays AS (
            SELECT
                i.subject_id,
                i.stay_id,
                i.intime,
                i.first_careunit,
                p.anchor_age,
                ROW_NUMBER() OVER (
                    PARTITION BY i.subject_id
                    ORDER BY i.intime ASC
                ) AS rn
            FROM mimic.icustays i
            JOIN mimic.patients p
            ON i.subject_id = p.subject_id
        )
    """

    from_stay = """
        FROM ranked_stays
        WHERE rn = 1
        AND anchor_age >= 18
        AND stay_id IN (
            SELECT stay_id
            FROM hrl.sepsis3
        )
        AND first_careunit IN (
            'Medical Intensive Care Unit (MICU)',
            'Surgical Intensive Care Unit (SICU)',
            'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Intensive Care Unit (ICU)'
        )
    """

    conn, cur = db.open_db()

    cohort = target_cohort(
        with_stay,
        from_stay,
        conn,
        cur
    )

    stayids = cohort.query()

    states = states_preprocessor(
        conn,
        cur,
        interval,
        my_required_items,
        stayids,
        initial_values,
        zero_fill_cols
    )

    df_query = states.main()

    sofa_module = sofa(
        conn,
        cur,
        stayids
    )

    df_sofa = sofa_module.main()

    survival_module = survival_labeler(
        conn,
        cur,
        stayids
    )

    df_survival = survival_module.main()

    stay_str = ','.join(
        map(str, stayids)
    )

    q_gcs = f"""
        SELECT
            stay_id,
            time_hour AS charttime,
            gcs_score
        FROM hrl.gcs
        WHERE stay_id IN ({stay_str})
    """

    df_gcs = pd.read_sql(
        q_gcs,
        conn
    )

    df_gcs['charttime'] = pd.to_datetime(
        df_gcs['charttime']
    )

    df_query['charttime'] = pd.to_datetime(
        df_query['charttime']
    )

    df_query = pd.merge(
        df_query,
        df_gcs,
        on=[
            'stay_id',
            'charttime'
        ],
        how='left'
    )

    df_query['gcs_score'] = (
        df_query
        .groupby('stay_id')['gcs_score']
        .ffill()
        .bfill()
        .fillna(15)
    )

    df_sofa = df_sofa.rename(
        columns={
            'chart_hour': 'charttime',
            'total_sofa_score': 'sofa_score'
        }
    )

    df_sofa['charttime'] = pd.to_datetime(
        df_sofa['charttime']
    )

    df_merged = pd.merge(
        df_query,
        df_sofa[
            [
                'stay_id',
                'charttime',
                'sofa_score'
            ]
        ],
        on=[
            'stay_id',
            'charttime'
        ],
        how='left'
    )

    df_merged = (
        df_merged
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    df_merged['sofa_score'] = (
        df_merged
        .groupby('stay_id')['sofa_score']
        .ffill()
        .bfill()
        .fillna(0)
    )

    onset_mask = (
        (df_merged['SIRS'] >= 2)
        |
        (df_merged['sofa_score'] >= 2)
    )

    onset_df = (
        df_merged[
            onset_mask
        ]
        .groupby('stay_id')['charttime']
        .min()
        .reset_index()
        .rename(
            columns={
                'charttime': 'onset_time'
            }
        )
    )

    df_merged = pd.merge(
        df_merged,
        onset_df,
        on='stay_id',
        how='inner'
    )

    df_merged['hours_from_onset'] = (
        (
            df_merged['charttime']
            - df_merged['onset_time']
        )
        .dt.total_seconds()
        / 3600
    )

    if (
        'Lactate_chart' in df_merged.columns
        and
        'Lactate' in df_merged.columns
    ):
        df_merged['Lactate'] = (
            df_merged['Lactate']
            .fillna(
                df_merged['Lactate_chart']
            )
        )

        df_merged = df_merged.drop(
            columns=[
                'Lactate_chart'
            ]
        )

    col_mapping = {
        'PaO2': 'pao2',
        'FiO2': 'fio2',
        'platelets_valuenum': 'platelets',
        'tb_valuenum': 'bilirubin',
        'creatinine_valuenum': 'creatinine',
        'Lactate': 'lactate',
        'gcs_score': 'gcs'
    }

    df_merged = df_merged.rename(
        columns=col_mapping
    )

    df_merged = pd.merge(
        df_merged,
        df_survival,
        on='stay_id',
        how='left'
    )

    df_merged = (
        df_merged
        .dropna(
            subset=[
                'survival'
            ]
        )
        .copy()
    )

    df_merged['survival'] = (
        df_merged['survival']
        .astype(int)
    )

    state_grid = (
        df_merged[
            [
                'stay_id',
                'charttime'
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    action_module = action_preprocessor(
        conn,
        cur,
        interval,
        stayids
    )

    df_actions = action_module.main(
        state_grid
    )

    if not df_actions.empty:

        df_actions['charttime'] = pd.to_datetime(
            df_actions['charttime']
        )

        df_full = pd.merge(
            df_merged,
            df_actions,
            on=[
                'stay_id',
                'charttime'
            ],
            how='left'
        )

        df_full['iv_fluid'] = (
            df_full['iv_fluid']
            .fillna(0.0)
        )

        df_full['vaso'] = (
            df_full['vaso']
            .fillna(0.0)
        )

        df_full['iv_action'] = (
            df_full['iv_action']
            .fillna(1)
            .astype(int)
        )

        df_full['vaso_action'] = (
            df_full['vaso_action']
            .fillna(1)
            .astype(int)
        )

        df_full['final_action'] = (
            df_full['final_action']
            .fillna(0)
            .astype(int)
        )

    else:

        df_full = df_merged.copy()

        df_full['iv_fluid'] = 0.0
        df_full['vaso'] = 0.0
        df_full['iv_action'] = 1
        df_full['vaso_action'] = 1
        df_full['final_action'] = 0

    df_full = (
        df_full
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\n전체 Action distribution"
    )

    print(
        df_full[
            'final_action'
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    df_24h = (
        df_full[
            (
                df_full['hours_from_onset'] >= 0
            )
            &
            (
                df_full['hours_from_onset'] <= 24
            )
        ]
        .copy()
    )

    df_24h = (
        df_24h
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    sampler = Sample(
        data=df_query,
        iterations=1000,
        threshold=0.05,
        sample_size=1000,
        sofa=df_24h
    )

    df_sampled = sampler.main()

    df_sampled = df_sampled.rename(
        columns=col_mapping
    )

    df_sampled['charttime'] = pd.to_datetime(
        df_sampled['charttime']
    )

    sampled_stay_ids = (
        df_sampled[
            'stay_id'
        ]
        .unique()
        .copy()
    )

    df_sampled = pd.merge(
        df_sampled,
        onset_df,
        on='stay_id',
        how='inner'
    )

    df_sampled['hours_from_onset'] = (
        (
            df_sampled['charttime']
            - df_sampled['onset_time']
        )
        .dt.total_seconds()
        / 3600
    )

    df_sampled = (
        df_sampled[
            (
                df_sampled['hours_from_onset'] >= 0
            )
            &
            (
                df_sampled['hours_from_onset'] <= 24
            )
        ]
        .copy()
    )

    df_sampled = (
        df_sampled
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\nSampled patients:",
        len(sampled_stay_ids)
    )

    print(
        "Encoder 0~24h rows:",
        len(df_sampled)
    )

    features_col = [
        'pao2',
        'fio2',
        'platelets',
        'bilirubin',
        'creatinine',
        'lactate',
        'gcs'
    ]

    df_sampled[
        features_col
    ] = (
        df_sampled
        .groupby('stay_id')[
            features_col
        ]
        .ffill()
        .bfill()
        .fillna(0)
    )

    hmm_module = SepsisHMM(
        n_components=4
    )

    hmm_module.train(
        df_sampled,
        features_col
    )

    hmm_module.save_model()

    df_sampled['hmm_state'] = (
        hmm_module.predict(
            df_sampled,
            features_col
        )
    )

    next_features_cols = [
        f"{c}_next"
        for c in features_col
    ]

    df_sampled[
        next_features_cols
    ] = (
        df_sampled
        .groupby('stay_id')[
            features_col
        ]
        .shift(-1)
    )

    df_shifted = (
        df_sampled
        .dropna(
            subset=
                next_features_cols
        )
        .copy()
    )

    encoder_stay_ids = (
        df_shifted[
            'stay_id'
        ]
        .unique()
        .copy()
    )

    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        encoder_stay_ids
    )

    encoder_split = int(
        len(
            encoder_stay_ids
        ) * 0.8
    )

    encoder_train_ids = (
        encoder_stay_ids[
            :encoder_split
        ]
    )

    encoder_val_ids = (
        encoder_stay_ids[
            encoder_split:
        ]
    )

    encoder_train_df = (
        df_shifted[
            df_shifted[
                'stay_id'
            ]
            .isin(
                encoder_train_ids
            )
        ]
        .copy()
    )

    encoder_val_df = (
        df_shifted[
            df_shifted[
                'stay_id'
            ]
            .isin(
                encoder_val_ids
            )
        ]
        .copy()
    )

    X_train = (
        hmm_module.scaler
        .transform(
            encoder_train_df[
                features_col
            ].values
        )
    )

    X_train_next = (
        hmm_module.scaler
        .transform(
            encoder_train_df[
                next_features_cols
            ].values
        )
    )

    X_val = (
        hmm_module.scaler
        .transform(
            encoder_val_df[
                features_col
            ].values
        )
    )

    X_val_next = (
        hmm_module.scaler
        .transform(
            encoder_val_df[
                next_features_cols
            ].values
        )
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(
            X_train
        ),
        torch.FloatTensor(
            X_train_next
        ),
        torch.LongTensor(
            encoder_train_df[
                'hmm_state'
            ].values
        )
    )

    val_dataset = TensorDataset(
        torch.FloatTensor(
            X_val
        ),
        torch.FloatTensor(
            X_val_next
        ),
        torch.LongTensor(
            encoder_val_df[
                'hmm_state'
            ].values
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )

    encoder_module = SepsisEncoder(
        input_dim=
            len(features_col),
        latent_dim=7,
        device=str(device)
    )

    encoder_module.train(
        train_loader=
            train_dataloader,
        val_loader=
            val_dataloader,
        epochs=20
    )

    encoder_module.build_distributions(
        train_dataloader
    )

    encoder_module.save_model(
        path=
            'sde_encoder_dict.pth'
    )

    encoder_module.encoder.eval()
    encoder_module.sde.eval()

    for param in (
        encoder_module
        .encoder
        .parameters()
    ):
        param.requires_grad = False

    for param in (
        encoder_module
        .sde
        .parameters()
    ):
        param.requires_grad = False

    df_rl = (
        df_full[
            ~df_full[
                'stay_id'
            ]
            .isin(
                sampled_stay_ids
            )
        ]
        .copy()
    )

    df_rl = (
        df_rl
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\n전체 patients:",
        df_full[
            'stay_id'
        ].nunique()
    )

    print(
        "Encoder sampled patients:",
        len(
            sampled_stay_ids
        )
    )

    print(
        "RL patients:",
        df_rl[
            'stay_id'
        ].nunique()
    )

    overlap = (
        set(
            sampled_stay_ids
        )
        &
        set(
            df_rl[
                'stay_id'
            ].unique()
        )
    )

    print(
        "Encoder/RL overlap:",
        len(overlap)
    )

    if len(overlap) > 0:
        raise ValueError(
            "Encoder sampled patients and RL patients overlap."
        )

    df_rl[
        features_col
    ] = (
        df_rl
        .groupby('stay_id')[
            features_col
        ]
        .ffill()
        .bfill()
        .fillna(0)
    )

    df_rl[
        next_features_cols
    ] = (
        df_rl
        .groupby('stay_id')[
            features_col
        ]
        .shift(-1)
    )

    df_rl[
        'sofa_score_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'sofa_score'
        ]
        .shift(-1)
    )

    df_rl[
        'next_charttime'
    ] = (
        df_rl
        .groupby('stay_id')[
            'charttime'
        ]
        .shift(-1)
    )

    df_rl[
        'action_to_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'final_action'
        ]
        .shift(-1)
    )

    df_rl[
        'iv_action_to_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'iv_action'
        ]
        .shift(-1)
    )

    df_rl[
        'vaso_action_to_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'vaso_action'
        ]
        .shift(-1)
    )

    df_rl[
        'iv_fluid_to_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'iv_fluid'
        ]
        .shift(-1)
    )

    df_rl[
        'vaso_to_next'
    ] = (
        df_rl
        .groupby('stay_id')[
            'vaso'
        ]
        .shift(-1)
    )

    df_rl_shifted = (
        df_rl
        .dropna(
            subset=
                next_features_cols
                + [
                    'sofa_score_next',
                    'next_charttime',
                    'action_to_next'
                ]
        )
        .copy()
    )

    df_rl_shifted[
        'action_to_next'
    ] = (
        df_rl_shifted[
            'action_to_next'
        ]
        .astype(int)
    )

    df_rl_shifted[
        'iv_action_to_next'
    ] = (
        df_rl_shifted[
            'iv_action_to_next'
        ]
        .astype(int)
    )

    df_rl_shifted[
        'vaso_action_to_next'
    ] = (
        df_rl_shifted[
            'vaso_action_to_next'
        ]
        .astype(int)
    )

    df_rl_shifted = (
        df_rl_shifted
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(drop=True)
    )

    df_rl_shifted[
        'is_last'
    ] = (
        df_rl_shifted
        .groupby('stay_id')
        .cumcount(
            ascending=False
        )
        == 0
    )

    if not (
        df_rl_shifted[
            'action_to_next'
        ]
        .between(
            0,
            24
        )
        .all()
    ):
        raise ValueError(
            "action_to_next must be between 0 and 24"
        )

    print(
        "\nRL 전체 기간 범위"
    )

    print(
        df_rl_shifted[
            'hours_from_onset'
        ].min(),
        df_rl_shifted[
            'hours_from_onset'
        ].max()
    )

    print(
        "\nRL transition example"
    )

    print(
        df_rl_shifted[
            [
                'stay_id',
                'charttime',
                'next_charttime',
                'hours_from_onset',
                'iv_fluid_to_next',
                'vaso_to_next',
                'action_to_next',
                'sofa_score',
                'sofa_score_next'
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print(
        "\nRL Action distribution"
    )

    print(
        df_rl_shifted[
            'action_to_next'
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    unique_rl_stay_ids = (
        df_rl_shifted[
            'stay_id'
        ]
        .unique()
        .copy()
    )

    rng.shuffle(
        unique_rl_stay_ids
    )

    n_rl = len(
        unique_rl_stay_ids
    )

    if n_rl < 3:
        raise ValueError(
            "Not enough RL stays"
        )

    train_end = max(
        1,
        int(
            n_rl * 0.70
        )
    )

    val_end = max(
        train_end + 1,
        int(
            n_rl * 0.85
        )
    )

    if val_end >= n_rl:
        val_end = (
            n_rl - 1
        )

    train_stay_ids = (
        unique_rl_stay_ids[
            :train_end
        ]
    )

    val_stay_ids = (
        unique_rl_stay_ids[
            train_end:
            val_end
        ]
    )

    test_stay_ids = (
        unique_rl_stay_ids[
            val_end:
        ]
    )

    df_rl_train = (
        df_rl_shifted[
            df_rl_shifted[
                'stay_id'
            ]
            .isin(
                train_stay_ids
            )
        ]
        .copy()
    )

    df_rl_val = (
        df_rl_shifted[
            df_rl_shifted[
                'stay_id'
            ]
            .isin(
                val_stay_ids
            )
        ]
        .copy()
    )

    df_rl_test = (
        df_rl_shifted[
            df_rl_shifted[
                'stay_id'
            ]
            .isin(
                test_stay_ids
            )
        ]
        .copy()
    )

    print(
        "\nRL stays"
    )

    print(
        "Train:",
        len(
            train_stay_ids
        )
    )

    print(
        "Validation:",
        len(
            val_stay_ids
        )
    )

    print(
        "Test:",
        len(
            test_stay_ids
        )
    )

    print(
        "\nRL transitions"
    )

    print(
        "Train:",
        len(
            df_rl_train
        )
    )

    print(
        "Validation:",
        len(
            df_rl_val
        )
    )

    print(
        "Test:",
        len(
            df_rl_test
        )
    )

    rl_train_dataset = (
        make_rl_dataset(
            df_rl_train,
            features_col,
            next_features_cols,
            hmm_module.scaler
        )
    )

    rl_val_dataset = (
        make_rl_dataset(
            df_rl_val,
            features_col,
            next_features_cols,
            hmm_module.scaler
        )
    )

    rl_test_dataset = (
        make_rl_dataset(
            df_rl_test,
            features_col,
            next_features_cols,
            hmm_module.scaler
        )
    )

    rl_train_dataloader = DataLoader(
        rl_train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False
    )

    rl_val_dataloader = DataLoader(
        rl_val_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )

    rl_test_dataloader = DataLoader(
        rl_test_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False
    )

    q_net = LowLevelQNetwork(
        latent_dim=7,
        action_dim=25
    ).to(device)

    p_net = LowLevelQNetwork(
        latent_dim=7,
        action_dim=25
    ).to(device)

    meta_net = HighLevelPolicy(
        latent_dim=7,
        num_policies=2
    ).to(device)

    target_q = (
        copy.deepcopy(
            q_net
        )
        .to(device)
    )

    target_p = (
        copy.deepcopy(
            p_net
        )
        .to(device)
    )

    target_q.eval()
    target_p.eval()

    for param in (
        target_q.parameters()
    ):
        param.requires_grad = False

    for param in (
        target_p.parameters()
    ):
        param.requires_grad = False

    opt_q = optim.Adam(
        q_net.parameters(),
        lr=3e-4
    )

    opt_p = optim.Adam(
        p_net.parameters(),
        lr=3e-4
    )

    opt_meta = optim.Adam(
        meta_net.parameters(),
        lr=3e-4
    )

    g_mean, g_std = (
        compute_uncertainty_stats(
            rl_train_dataloader,
            encoder_module,
            device
        )
    )

    print(
        "\nUncertainty mean:",
        g_mean
    )

    print(
        "Uncertainty std:",
        g_std
    )

    best_val_loss = float(
        'inf'
    )

    for epoch in range(
        rl_epochs
    ):

        q_net.train()
        p_net.train()
        meta_net.train()

        total_q_loss = 0.0
        total_p_loss = 0.0
        total_meta_loss = 0.0

        total_td_q = 0.0
        total_td_p = 0.0

        total_cql_q = 0.0
        total_cql_p = 0.0

        total_samples = 0

        hierarchy_correct = 0
        q_correct = 0
        p_correct = 0

        prospect_count = 0

        num_batches = 0

        for (
            batch_X,
            batch_X_next,
            batch_act,
            batch_sofa,
            batch_sofa_next,
            batch_surv,
            batch_last
        ) in rl_train_dataloader:

            batch_X = batch_X.to(
                device
            )

            batch_X_next = (
                batch_X_next.to(
                    device
                )
            )

            batch_act = (
                batch_act.to(
                    device
                )
            )

            batch_sofa = (
                batch_sofa.to(
                    device
                )
            )

            batch_sofa_next = (
                batch_sofa_next.to(
                    device
                )
            )

            batch_surv = (
                batch_surv.to(
                    device
                )
            )

            batch_last = (
                batch_last.to(
                    device
                )
            )

            with torch.no_grad():

                z_curr = get_latent(
                    encoder_module,
                    batch_X
                )

                z_next = get_latent(
                    encoder_module,
                    batch_X_next
                )

                g_norm = get_uncertainty(
                    encoder_module,
                    z_curr
                )

                g_scaled = scale_uncertainty(
                    g_norm,
                    g_mean,
                    g_std
                )

            reward_q, reward_p = (
                make_rewards(
                    batch_sofa,
                    batch_sofa_next,
                    batch_surv,
                    batch_last,
                    lambda_pt
                )
            )

            q_all = q_net(
                z_curr
            )

            q_data = (
                q_all
                .gather(
                    1,
                    batch_act.unsqueeze(
                        1
                    )
                )
                .squeeze(1)
            )

            with torch.no_grad():

                next_action_q = (
                    q_net(
                        z_next
                    )
                    .argmax(
                        dim=1,
                        keepdim=True
                    )
                )

                next_q = (
                    target_q(
                        z_next
                    )
                    .gather(
                        1,
                        next_action_q
                    )
                    .squeeze(1)
                )

                td_target_q = (
                    reward_q
                    + gamma
                    * next_q
                    * (
                        ~batch_last
                    ).float()
                )

            loss_td_q = (
                F.smooth_l1_loss(
                    q_data,
                    td_target_q
                )
            )

            loss_cql_q = (
                cql_loss(
                    q_all,
                    q_data
                )
            )

            loss_q = (
                loss_td_q
                + cql_weight
                * loss_cql_q
            )

            opt_q.zero_grad()

            loss_q.backward()

            torch.nn.utils.clip_grad_norm_(
                q_net.parameters(),
                1.0
            )

            opt_q.step()

            p_all = p_net(
                z_curr
            )

            p_data = (
                p_all
                .gather(
                    1,
                    batch_act.unsqueeze(
                        1
                    )
                )
                .squeeze(1)
            )

            with torch.no_grad():

                next_action_p = (
                    p_net(
                        z_next
                    )
                    .argmax(
                        dim=1,
                        keepdim=True
                    )
                )

                next_p = (
                    target_p(
                        z_next
                    )
                    .gather(
                        1,
                        next_action_p
                    )
                    .squeeze(1)
                )

                td_target_p = (
                    reward_p
                    + gamma
                    * next_p
                    * (
                        ~batch_last
                    ).float()
                )

            loss_td_p = (
                F.smooth_l1_loss(
                    p_data,
                    td_target_p
                )
            )

            loss_cql_p = (
                cql_loss(
                    p_all,
                    p_data
                )
            )

            loss_p = (
                loss_td_p
                + cql_weight
                * loss_cql_p
            )

            opt_p.zero_grad()

            loss_p.backward()

            torch.nn.utils.clip_grad_norm_(
                p_net.parameters(),
                1.0
            )

            opt_p.step()

            with torch.no_grad():

                q_meta = q_net(
                    z_curr
                )

                p_meta = p_net(
                    z_curr
                )

            probs = meta_net(
                z_curr,
                g_scaled
            )

            (
                loss_meta,
                _,
                _,
                _
            ) = meta_net.compute_loss(
                probs,
                q_meta,
                p_meta,
                g_scaled,
                alpha=0.25,
                beta=0.01
            )

            opt_meta.zero_grad()

            loss_meta.backward()

            torch.nn.utils.clip_grad_norm_(
                meta_net.parameters(),
                1.0
            )

            opt_meta.step()

            soft_update(
                target_q,
                q_net,
                tau
            )

            soft_update(
                target_p,
                p_net,
                tau
            )

            with torch.no_grad():

                probs_eval = meta_net(
                    z_curr,
                    g_scaled
                )

                meta_choice = (
                    probs_eval
                    .argmax(
                        dim=1
                    )
                )

                q_action = (
                    q_net(
                        z_curr
                    )
                    .argmax(
                        dim=1
                    )
                )

                p_action = (
                    p_net(
                        z_curr
                    )
                    .argmax(
                        dim=1
                    )
                )

                hierarchy_action = (
                    torch.where(
                        meta_choice == 0,
                        q_action,
                        p_action
                    )
                )

                batch_size = (
                    batch_act.size(
                        0
                    )
                )

                total_samples += (
                    batch_size
                )

                hierarchy_correct += (
                    hierarchy_action
                    == batch_act
                ).sum().item()

                q_correct += (
                    q_action
                    == batch_act
                ).sum().item()

                p_correct += (
                    p_action
                    == batch_act
                ).sum().item()

                prospect_count += (
                    meta_choice
                    == 1
                ).sum().item()

            total_q_loss += (
                loss_q.item()
            )

            total_p_loss += (
                loss_p.item()
            )

            total_meta_loss += (
                loss_meta.item()
            )

            total_td_q += (
                loss_td_q.item()
            )

            total_td_p += (
                loss_td_p.item()
            )

            total_cql_q += (
                loss_cql_q.item()
            )

            total_cql_p += (
                loss_cql_p.item()
            )

            num_batches += 1

        train_hierarchy_agreement = (
            hierarchy_correct
            / total_samples
        )

        train_q_agreement = (
            q_correct
            / total_samples
        )

        train_p_agreement = (
            p_correct
            / total_samples
        )

        train_prospect_ratio = (
            prospect_count
            / total_samples
        )

        val_results = evaluate(
            rl_val_dataloader,
            q_net,
            p_net,
            meta_net,
            target_q,
            target_p,
            encoder_module,
            device,
            gamma,
            lambda_pt,
            cql_weight,
            g_mean,
            g_std
        )

        print(
            f"\nEpoch {epoch + 1:03d}"
        )

        print(
            f"Train Q Loss: "
            f"{total_q_loss / num_batches:.4f}"
            f" | TD: "
            f"{total_td_q / num_batches:.4f}"
            f" | CQL: "
            f"{total_cql_q / num_batches:.4f}"
        )

        print(
            f"Train P Loss: "
            f"{total_p_loss / num_batches:.4f}"
            f" | TD: "
            f"{total_td_p / num_batches:.4f}"
            f" | CQL: "
            f"{total_cql_p / num_batches:.4f}"
        )

        print(
            f"Train Meta Loss: "
            f"{total_meta_loss / num_batches:.4f}"
        )

        print(
            f"Train Hierarchy Agreement: "
            f"{train_hierarchy_agreement:.4f}"
            f" | Q: "
            f"{train_q_agreement:.4f}"
            f" | Prospect: "
            f"{train_p_agreement:.4f}"
            f" | Prospect Ratio: "
            f"{train_prospect_ratio:.4f}"
        )

        print(
            f"Validation Total Loss: "
            f"{val_results['total_loss']:.4f}"
            f" | Q: "
            f"{val_results['q_loss']:.4f}"
            f" | P: "
            f"{val_results['p_loss']:.4f}"
            f" | Meta: "
            f"{val_results['meta_loss']:.4f}"
        )

        print(
            f"Validation Hierarchy Agreement: "
            f"{val_results['hierarchy_agreement']:.4f}"
            f" | Q: "
            f"{val_results['q_agreement']:.4f}"
            f" | Prospect: "
            f"{val_results['p_agreement']:.4f}"
            f" | Prospect Ratio: "
            f"{val_results['prospect_ratio']:.4f}"
        )

        if (
            val_results[
                'total_loss'
            ]
            < best_val_loss
        ):

            best_val_loss = (
                val_results[
                    'total_loss'
                ]
            )

            torch.save(
                {
                    'epoch':
                        epoch + 1,

                    'q_net':
                        q_net.state_dict(),

                    'p_net':
                        p_net.state_dict(),

                    'meta_net':
                        meta_net.state_dict(),

                    'target_q':
                        target_q.state_dict(),

                    'target_p':
                        target_p.state_dict(),

                    'val_loss':
                        best_val_loss,

                    'g_mean':
                        g_mean,

                    'g_std':
                        g_std,

                    'gamma':
                        gamma,

                    'lambda_pt':
                        lambda_pt,

                    'cql_weight':
                        cql_weight
                },
                'hrl_best_agents.pth'
            )

    checkpoint = torch.load(
        'hrl_best_agents.pth',
        map_location=device
    )

    q_net.load_state_dict(
        checkpoint[
            'q_net'
        ]
    )

    p_net.load_state_dict(
        checkpoint[
            'p_net'
        ]
    )

    meta_net.load_state_dict(
        checkpoint[
            'meta_net'
        ]
    )

    target_q.load_state_dict(
        checkpoint[
            'target_q'
        ]
    )

    target_p.load_state_dict(
        checkpoint[
            'target_p'
        ]
    )

    g_mean = (
        checkpoint[
            'g_mean'
        ]
    )

    g_std = (
        checkpoint[
            'g_std'
        ]
    )

    test_results = evaluate(
        rl_test_dataloader,
        q_net,
        p_net,
        meta_net,
        target_q,
        target_p,
        encoder_module,
        device,
        gamma,
        lambda_pt,
        cql_weight,
        g_mean,
        g_std
    )

    print(
        "\nBest Epoch:",
        checkpoint[
            'epoch'
        ]
    )

    print(
        "Best Validation Loss:",
        checkpoint[
            'val_loss'
        ]
    )

    print(
        "\nTest Total Loss:",
        test_results[
            'total_loss'
        ]
    )

    print(
        "Test Q Loss:",
        test_results[
            'q_loss'
        ]
    )

    print(
        "Test Prospect Loss:",
        test_results[
            'p_loss'
        ]
    )

    print(
        "Test Meta Loss:",
        test_results[
            'meta_loss'
        ]
    )

    print(
        "Test Hierarchy Agreement:",
        test_results[
            'hierarchy_agreement'
        ]
    )

    print(
        "Test Q Agreement:",
        test_results[
            'q_agreement'
        ]
    )

    print(
        "Test Prospect Agreement:",
        test_results[
            'p_agreement'
        ]
    )

    print(
        "Test Prospect Selection Ratio:",
        test_results[
            'prospect_ratio'
        ]
    )