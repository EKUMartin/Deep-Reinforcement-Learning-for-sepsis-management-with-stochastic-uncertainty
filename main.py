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
from Visualization.results import visualizer
from Pretraining.encoder import SepsisEncoder
from Pretraining.hmm import SepsisHMM
from Pretraining.sampling import Sample

from policy.policy import (
    LowLevelQNetwork,
    HighLevelQNetwork
)

from db_conn import db


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(
    target,
    source,
    tau
):
    for target_param, source_param in zip(
        target.parameters(),
        source.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data
            + (1.0 - tau)
            * target_param.data
        )


def get_latent(
    encoder_module,
    x
):
    mu, _ = (
        encoder_module
        .encoder(x)
    )

    return mu


def get_uncertainty(
    encoder_module,
    z
):
    t = torch.zeros_like(
        z[:, :1]
    )

    ty = torch.cat(
        [
            t,
            z
        ],
        dim=1
    )

    g_val = (
        encoder_module
        .sde
        .g_net(ty)
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
        g_norm
        - g_mean
    ) / (
        g_std
        + 1e-6
    )

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
        sofa_curr
        - sofa_next
    )

    reward_step = (
        torch.clamp(
            reward_step,
            min=-4.0,
            max=4.0
        )
        / 4.0
    )

    terminal_reward = (
        torch.where(
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
    )

    reward_q = (
        reward_step
        + is_last.float()
        * terminal_reward
    )

    reward_p = torch.where(
        reward_q >= 0,
        reward_q,
        lambda_pt
        * reward_q
    )

    return (
        reward_q,
        reward_p
    )


def make_high_level_reward(
    base_reward,
    option,
    uncertainty,
    uncertainty_reward_weight
):
    uncertainty_signal = (
        2.0
        * uncertainty
        - 1.0
    )

    option_signal = (
        2.0
        * option.float()
        - 1.0
    )

    alignment = (
        option_signal
        * uncertainty_signal
    )

    reward_high = (
        base_reward
        + uncertainty_reward_weight
        * alignment
    )

    return (
        reward_high,
        alignment
    )


def cql_loss(
    q_all,
    q_data
):
    return (
        torch.logsumexp(
            q_all,
            dim=1
        ).mean()
        - q_data.mean()
    )


def get_option_margin(
    q_values,
    p_values,
    behavior_action
):
    q_mean = q_values.mean(
        dim=1,
        keepdim=True
    )

    q_std = q_values.std(
        dim=1,
        keepdim=True,
        unbiased=False
    ) + 1e-6

    p_mean = p_values.mean(
        dim=1,
        keepdim=True
    )

    p_std = p_values.std(
        dim=1,
        keepdim=True,
        unbiased=False
    ) + 1e-6

    q_normalized = (
        q_values - q_mean
    ) / q_std

    p_normalized = (
        p_values - p_mean
    ) / p_std

    q_log_prob = (
        F.log_softmax(
            q_normalized,
            dim=1
        )
    )

    p_log_prob = (
        F.log_softmax(
            p_normalized,
            dim=1
        )
    )

    q_score = (
        q_log_prob
        .gather(
            1,
            behavior_action.unsqueeze(1)
        )
        .squeeze(1)
    )

    p_score = (
        p_log_prob
        .gather(
            1,
            behavior_action.unsqueeze(1)
        )
        .squeeze(1)
    )

    margin = (
        p_score
        - q_score
    )

    return margin


def compute_option_margin_stats(
    loader,
    encoder_module,
    q_net,
    p_net,
    device
):
    margins = []

    encoder_module.encoder.eval()

    q_net.eval()
    p_net.eval()

    with torch.no_grad():

        for (
            batch_X,
            _,
            batch_act,
            _,
            _,
            _,
            _
        ) in loader:

            batch_X = (
                batch_X.to(
                    device
                )
            )

            batch_act = (
                batch_act.to(
                    device
                )
            )

            z = get_latent(
                encoder_module,
                batch_X
            )

            q_values = (
                q_net(z)
            )

            p_values = (
                p_net(z)
            )

            margin = (
                get_option_margin(
                    q_values,
                    p_values,
                    batch_act
                )
            )

            margins.append(
                margin
                .cpu()
            )

    margins = torch.cat(
        margins,
        dim=0
    )

    margin_mean = (
        margins.mean().item()
    )

    margin_std = (
        margins.std(
            unbiased=False
        ).item()
    )

    return (
        margin_mean,
        max(
            margin_std,
            1e-6
        )
    )


def infer_high_level_option(
    q_values,
    p_values,
    behavior_action,
    uncertainty,
    margin_mean,
    margin_std,
    behavior_option_weight=1.0,
    uncertainty_option_weight=1.0
):
    margin = (
        get_option_margin(
            q_values,
            p_values,
            behavior_action
        )
    )

    margin_z = (
        margin
        - margin_mean
    ) / (
        margin_std
        + 1e-6
    )

    uncertainty_signal = (
        2.0
        * uncertainty
        - 1.0
    )

    option_score = (
        behavior_option_weight
        * margin_z
        +
        uncertainty_option_weight
        * uncertainty_signal
    )

    option = (
        option_score
        > 0
    ).long()

    return (
        option,
        margin_z,
        uncertainty_signal,
        option_score
    )


def make_rl_dataset(
    df,
    features_col,
    next_features_cols,
    scaler
):
    x_curr = (
        scaler.transform(
            df[
                features_col
            ].values
        )
    )

    x_next = (
        scaler.transform(
            df[
                next_features_cols
            ].values
        )
    )

    return TensorDataset(
        torch.FloatTensor(
            x_curr
        ),
        torch.FloatTensor(
            x_next
        ),
        torch.LongTensor(
            df[
                'action_to_next'
            ].values
        ),
        torch.FloatTensor(
            df[
                'sofa_score'
            ].values
        ),
        torch.FloatTensor(
            df[
                'sofa_score_next'
            ].values
        ),
        torch.LongTensor(
            df[
                'survival'
            ].values
        ),
        torch.BoolTensor(
            df[
                'is_last'
            ].values
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

            x = (
                batch[0]
                .to(device)
            )

            z = get_latent(
                encoder_module,
                x
            )

            g = get_uncertainty(
                encoder_module,
                z
            )

            values.append(
                g.cpu()
            )

    values = torch.cat(
        values,
        dim=0
    )

    mean = (
        values.mean().item()
    )

    std = (
        values.std(
            unbiased=False
        ).item()
    )

    return (
        mean,
        max(
            std,
            1e-6
        )
    )


def evaluate_low_level(
    loader,
    q_net,
    p_net,
    target_q,
    target_p,
    encoder_module,
    device,
    gamma,
    lambda_pt,
    cql_weight
):
    q_net.eval()
    p_net.eval()
    target_q.eval()
    target_p.eval()

    total_q_loss = 0.0
    total_p_loss = 0.0

    total_q_td = 0.0
    total_p_td = 0.0

    total_q_cql = 0.0
    total_p_cql = 0.0

    q_correct = 0
    p_correct = 0

    q_nonzero_correct = 0
    p_nonzero_correct = 0

    nonzero_total = 0
    total = 0
    batches = 0

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

            batch_X = (
                batch_X.to(
                    device
                )
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

            z_curr = get_latent(
                encoder_module,
                batch_X
            )

            z_next = get_latent(
                encoder_module,
                batch_X_next
            )

            (
                reward_q,
                reward_p
            ) = make_rewards(
                batch_sofa,
                batch_sofa_next,
                batch_surv,
                batch_last,
                lambda_pt
            )

            q_all = q_net(
                z_curr
            )

            q_data = (
                q_all
                .gather(
                    1,
                    batch_act
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            next_q_action = (
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
                    next_q_action
                )
                .squeeze(1)
            )

            q_target = (
                reward_q
                + gamma
                * next_q
                * (
                    ~batch_last
                ).float()
            )

            q_td = (
                F.smooth_l1_loss(
                    q_data,
                    q_target
                )
            )

            q_cql = cql_loss(
                q_all,
                q_data
            )

            q_loss = (
                q_td
                + cql_weight
                * q_cql
            )

            p_all = p_net(
                z_curr
            )

            p_data = (
                p_all
                .gather(
                    1,
                    batch_act
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            next_p_action = (
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
                    next_p_action
                )
                .squeeze(1)
            )

            p_target = (
                reward_p
                + gamma
                * next_p
                * (
                    ~batch_last
                ).float()
            )

            p_td = (
                F.smooth_l1_loss(
                    p_data,
                    p_target
                )
            )

            p_cql = cql_loss(
                p_all,
                p_data
            )

            p_loss = (
                p_td
                + cql_weight
                * p_cql
            )

            q_action = (
                q_all.argmax(
                    dim=1
                )
            )

            p_action = (
                p_all.argmax(
                    dim=1
                )
            )

            q_correct += (
                q_action
                == batch_act
            ).sum().item()

            p_correct += (
                p_action
                == batch_act
            ).sum().item()

            nonzero_mask = (
                batch_act != 0
            )

            if (
                nonzero_mask.any()
            ):

                q_nonzero_correct += (
                    q_action[
                        nonzero_mask
                    ]
                    ==
                    batch_act[
                        nonzero_mask
                    ]
                ).sum().item()

                p_nonzero_correct += (
                    p_action[
                        nonzero_mask
                    ]
                    ==
                    batch_act[
                        nonzero_mask
                    ]
                ).sum().item()

                nonzero_total += (
                    nonzero_mask
                    .sum()
                    .item()
                )

            total += (
                batch_act.size(0)
            )

            total_q_loss += (
                q_loss.item()
            )

            total_p_loss += (
                p_loss.item()
            )

            total_q_td += (
                q_td.item()
            )

            total_p_td += (
                p_td.item()
            )

            total_q_cql += (
                q_cql.item()
            )

            total_p_cql += (
                p_cql.item()
            )

            batches += 1

    return {
        'q_loss':
            total_q_loss
            / batches,

        'p_loss':
            total_p_loss
            / batches,

        'total_loss':
            (
                total_q_loss
                + total_p_loss
            )
            / batches,

        'q_td':
            total_q_td
            / batches,

        'p_td':
            total_p_td
            / batches,

        'q_cql':
            total_q_cql
            / batches,

        'p_cql':
            total_p_cql
            / batches,

        'q_agreement':
            q_correct
            / total,

        'p_agreement':
            p_correct
            / total,

        'q_nonzero_agreement':
            (
                q_nonzero_correct
                / nonzero_total
                if nonzero_total > 0
                else 0.0
            ),

        'p_nonzero_agreement':
            (
                p_nonzero_correct
                / nonzero_total
                if nonzero_total > 0
                else 0.0
            )
    }


def evaluate_high_level(
    loader,
    q_net,
    p_net,
    high_net,
    target_high,
    encoder_module,
    device,
    gamma,
    lambda_pt,
    high_cql_weight,
    g_mean,
    g_std,
    margin_mean,
    margin_std,
    behavior_option_weight,
    uncertainty_option_weight,
    uncertainty_reward_weight
):
    q_net.eval()
    p_net.eval()
    high_net.eval()
    target_high.eval()

    total_loss = 0.0
    total_td = 0.0
    total_cql = 0.0

    option_correct = 0
    hierarchy_correct = 0
    hierarchy_nonzero_correct = 0

    q_correct = 0
    p_correct = 0

    prospect_pred = 0
    prospect_behavior = 0

    action_zero_pred = 0

    total = 0
    nonzero_total = 0
    batches = 0

    pred_q_uncertainty_sum = 0.0
    pred_p_uncertainty_sum = 0.0

    pred_q_count = 0
    pred_p_count = 0

    behavior_q_uncertainty_sum = 0.0
    behavior_p_uncertainty_sum = 0.0

    behavior_q_count = 0
    behavior_p_count = 0

    alignment_sum = 0.0

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

            batch_X = (
                batch_X.to(
                    device
                )
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

            z_curr = get_latent(
                encoder_module,
                batch_X
            )

            z_next = get_latent(
                encoder_module,
                batch_X_next
            )

            g_curr = get_uncertainty(
                encoder_module,
                z_curr
            )

            g_next = get_uncertainty(
                encoder_module,
                z_next
            )

            g_curr = scale_uncertainty(
                g_curr,
                g_mean,
                g_std
            )

            g_next = scale_uncertainty(
                g_next,
                g_mean,
                g_std
            )

            (
                reward_q,
                _
            ) = make_rewards(
                batch_sofa,
                batch_sofa_next,
                batch_surv,
                batch_last,
                lambda_pt
            )

            q_values = q_net(
                z_curr
            )

            p_values = p_net(
                z_curr
            )

            (
                behavior_option,
                _,
                _,
                _
            ) = infer_high_level_option(
                q_values,
                p_values,
                batch_act,
                g_curr,
                margin_mean,
                margin_std,
                behavior_option_weight,
                uncertainty_option_weight
            )

            (
                reward_high,
                alignment
            ) = make_high_level_reward(
                reward_q,
                behavior_option,
                g_curr,
                uncertainty_reward_weight
            )

            high_q_all = (
                high_net(
                    z_curr,
                    g_curr
                )
            )

            high_q_data = (
                high_q_all
                .gather(
                    1,
                    behavior_option
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            next_option = (
                high_net(
                    z_next,
                    g_next
                )
                .argmax(
                    dim=1,
                    keepdim=True
                )
            )

            next_high_q = (
                target_high(
                    z_next,
                    g_next
                )
                .gather(
                    1,
                    next_option
                )
                .squeeze(1)
            )

            high_target = (
                reward_high
                + gamma
                * next_high_q
                * (
                    ~batch_last
                ).float()
            )

            high_td = (
                F.smooth_l1_loss(
                    high_q_data,
                    high_target
                )
            )

            high_cql = (
                cql_loss(
                    high_q_all,
                    high_q_data
                )
            )

            high_loss = (
                high_td
                + high_cql_weight
                * high_cql
            )

            pred_option = (
                high_q_all.argmax(
                    dim=1
                )
            )

            q_action = (
                q_values.argmax(
                    dim=1
                )
            )

            p_action = (
                p_values.argmax(
                    dim=1
                )
            )

            final_action = torch.where(
                pred_option == 0,
                q_action,
                p_action
            )

            option_correct += (
                pred_option
                == behavior_option
            ).sum().item()

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

            prospect_pred += (
                pred_option
                == 1
            ).sum().item()

            prospect_behavior += (
                behavior_option
                == 1
            ).sum().item()

            action_zero_pred += (
                final_action
                == 0
            ).sum().item()

            pred_q_mask = (
                pred_option == 0
            )

            pred_p_mask = (
                pred_option == 1
            )

            behavior_q_mask = (
                behavior_option == 0
            )

            behavior_p_mask = (
                behavior_option == 1
            )

            if pred_q_mask.any():

                pred_q_uncertainty_sum += (
                    g_curr[
                        pred_q_mask
                    ]
                    .sum()
                    .item()
                )

                pred_q_count += (
                    pred_q_mask
                    .sum()
                    .item()
                )

            if pred_p_mask.any():

                pred_p_uncertainty_sum += (
                    g_curr[
                        pred_p_mask
                    ]
                    .sum()
                    .item()
                )

                pred_p_count += (
                    pred_p_mask
                    .sum()
                    .item()
                )

            if behavior_q_mask.any():

                behavior_q_uncertainty_sum += (
                    g_curr[
                        behavior_q_mask
                    ]
                    .sum()
                    .item()
                )

                behavior_q_count += (
                    behavior_q_mask
                    .sum()
                    .item()
                )

            if behavior_p_mask.any():

                behavior_p_uncertainty_sum += (
                    g_curr[
                        behavior_p_mask
                    ]
                    .sum()
                    .item()
                )

                behavior_p_count += (
                    behavior_p_mask
                    .sum()
                    .item()
                )

            alignment_sum += (
                alignment
                .sum()
                .item()
            )

            nonzero_mask = (
                batch_act != 0
            )

            if nonzero_mask.any():

                hierarchy_nonzero_correct += (
                    final_action[
                        nonzero_mask
                    ]
                    ==
                    batch_act[
                        nonzero_mask
                    ]
                ).sum().item()

                nonzero_total += (
                    nonzero_mask
                    .sum()
                    .item()
                )

            total += (
                batch_act.size(0)
            )

            total_loss += (
                high_loss.item()
            )

            total_td += (
                high_td.item()
            )

            total_cql += (
                high_cql.item()
            )

            batches += 1

    pred_q_uncertainty = (
        pred_q_uncertainty_sum
        / pred_q_count
        if pred_q_count > 0
        else np.nan
    )

    pred_p_uncertainty = (
        pred_p_uncertainty_sum
        / pred_p_count
        if pred_p_count > 0
        else np.nan
    )

    behavior_q_uncertainty = (
        behavior_q_uncertainty_sum
        / behavior_q_count
        if behavior_q_count > 0
        else np.nan
    )

    behavior_p_uncertainty = (
        behavior_p_uncertainty_sum
        / behavior_p_count
        if behavior_p_count > 0
        else np.nan
    )

    return {
        'loss':
            total_loss
            / batches,

        'td':
            total_td
            / batches,

        'cql':
            total_cql
            / batches,

        'option_agreement':
            option_correct
            / total,

        'hierarchy_agreement':
            hierarchy_correct
            / total,

        'hierarchy_nonzero_agreement':
            (
                hierarchy_nonzero_correct
                / nonzero_total
                if nonzero_total > 0
                else 0.0
            ),

        'q_agreement':
            q_correct
            / total,

        'p_agreement':
            p_correct
            / total,

        'prospect_ratio':
            prospect_pred
            / total,

        'behavior_prospect_ratio':
            prospect_behavior
            / total,

        'predicted_action0_ratio':
            action_zero_pred
            / total,

        'pred_q_uncertainty':
            pred_q_uncertainty,

        'pred_prospect_uncertainty':
            pred_p_uncertainty,

        'behavior_q_uncertainty':
            behavior_q_uncertainty,

        'behavior_prospect_uncertainty':
            behavior_p_uncertainty,

        'uncertainty_gap':
            (
                pred_p_uncertainty
                - pred_q_uncertainty
                if (
                    not np.isnan(
                        pred_p_uncertainty
                    )
                    and
                    not np.isnan(
                        pred_q_uncertainty
                    )
                )
                else np.nan
            ),

        'mean_alignment':
            alignment_sum
            / total
    }


if __name__ == "__main__":

    set_seed(42)

    interval = 4
    analysis_window_hours = 96

    encoder_epochs = 20
    low_level_epochs = 100
    high_level_epochs = 100

    gamma = 0.99
    lambda_pt = 2.25

    tau = 0.005

    cql_weight = 0.5
    high_cql_weight = 0.1

    behavior_option_weight = 1.0

    uncertainty_option_weight = 1.0

    uncertainty_reward_weight = 0.25

    batch_size = 256

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        "Device:",
        device
    )

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

    conn, cur = (
        db.open_db()
    )

    cohort = target_cohort(
        with_stay,
        from_stay,
        conn,
        cur
    )

    stayids = (
        cohort.query()
    )

    states = states_preprocessor(
        conn,
        cur,
        interval,
        my_required_items,
        stayids,
        initial_values,
        zero_fill_cols
    )

    df_query = (
        states.main()
    )

    sofa_module = sofa(
        conn,
        cur,
        stayids
    )

    df_sofa = (
        sofa_module.main()
    )

    survival_module = (
        survival_labeler(
            conn,
            cur,
            stayids
        )
    )

    df_survival = (
        survival_module.main()
    )

    stay_str = (
        ','.join(
            map(
                str,
                stayids
            )
        )
    )

    df_icu_time = pd.read_sql(
        f"""
        SELECT
            stay_id,
            intime,
            outtime
        FROM mimic.icustays
        WHERE stay_id IN ({stay_str})
        """,
        conn
    )

    df_icu_time[
        'intime'
    ] = pd.to_datetime(
        df_icu_time[
            'intime'
        ]
    )

    df_icu_time[
        'outtime'
    ] = pd.to_datetime(
        df_icu_time[
            'outtime'
        ]
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

    df_gcs[
        'charttime'
    ] = pd.to_datetime(
        df_gcs[
            'charttime'
        ]
    )

    df_query[
        'charttime'
    ] = pd.to_datetime(
        df_query[
            'charttime'
        ]
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

    df_query[
        'gcs_score'
    ] = (
        df_query
        .groupby(
            'stay_id'
        )[
            'gcs_score'
        ]
        .ffill()
        .bfill()
        .fillna(15)
    )

    df_sofa = (
        df_sofa.rename(
            columns={
                'chart_hour':
                    'charttime',

                'total_sofa_score':
                    'sofa_score'
            }
        )
    )

    df_sofa[
        'charttime'
    ] = pd.to_datetime(
        df_sofa[
            'charttime'
        ]
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

    df_merged = pd.merge(
        df_merged,
        df_icu_time,
        on='stay_id',
        how='inner'
    )

    df_merged = (
        df_merged[
            (
                df_merged[
                    'charttime'
                ]
                >=
                df_merged[
                    'intime'
                ]
            )
            &
            (
                df_merged[
                    'charttime'
                ]
                <=
                df_merged[
                    'outtime'
                ]
            )
        ]
        .copy()
    )

    df_merged = (
        df_merged
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df_merged[
        'sofa_score'
    ] = (
        df_merged
        .groupby(
            'stay_id'
        )[
            'sofa_score'
        ]
        .ffill()
        .bfill()
        .fillna(0)
    )

    onset_mask = (
        (
            df_merged[
                'SIRS'
            ]
            >= 2
        )
        |
        (
            df_merged[
                'sofa_score'
            ]
            >= 2
        )
    )

    onset_df = (
        df_merged[
            onset_mask
        ]
        .groupby(
            'stay_id'
        )[
            'charttime'
        ]
        .min()
        .reset_index()
        .rename(
            columns={
                'charttime':
                    'onset_time'
            }
        )
    )

    df_merged = pd.merge(
        df_merged,
        onset_df,
        on='stay_id',
        how='inner'
    )

    df_merged[
        'hours_from_onset'
    ] = (
        (
            df_merged[
                'charttime'
            ]
            -
            df_merged[
                'onset_time'
            ]
        )
        .dt
        .total_seconds()
        / 3600
    )

    if (
        'Lactate_chart'
        in df_merged.columns
        and
        'Lactate'
        in df_merged.columns
    ):

        df_merged[
            'Lactate'
        ] = (
            df_merged[
                'Lactate'
            ]
            .fillna(
                df_merged[
                    'Lactate_chart'
                ]
            )
        )

        df_merged = (
            df_merged.drop(
                columns=[
                    'Lactate_chart'
                ]
            )
        )

    col_mapping = {
        'PaO2':
            'pao2',

        'FiO2':
            'fio2',

        'platelets_valuenum':
            'platelets',

        'tb_valuenum':
            'bilirubin',

        'creatinine_valuenum':
            'creatinine',

        'Lactate':
            'lactate',

        'gcs_score':
            'gcs'
    }

    df_merged = (
        df_merged.rename(
            columns=
                col_mapping
        )
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

    df_merged[
        'survival'
    ] = (
        df_merged[
            'survival'
        ]
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

    action_module = (
        action_preprocessor(
            conn,
            cur,
            interval,
            stayids
        )
    )

    df_actions = (
        action_module.main(
            state_grid
        )
    )

    if not df_actions.empty:

        df_actions[
            'charttime'
        ] = pd.to_datetime(
            df_actions[
                'charttime'
            ]
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

        df_full[
            'iv_fluid'
        ] = (
            df_full[
                'iv_fluid'
            ]
            .fillna(0.0)
        )

        df_full[
            'vaso'
        ] = (
            df_full[
                'vaso'
            ]
            .fillna(0.0)
        )

        df_full[
            'iv_action'
        ] = (
            df_full[
                'iv_action'
            ]
            .fillna(1)
            .astype(int)
        )

        df_full[
            'vaso_action'
        ] = (
            df_full[
                'vaso_action'
            ]
            .fillna(1)
            .astype(int)
        )

        df_full[
            'final_action'
        ] = (
            df_full[
                'final_action'
            ]
            .fillna(0)
            .astype(int)
        )

    else:

        df_full = (
            df_merged.copy()
        )

        df_full[
            'iv_fluid'
        ] = 0.0

        df_full[
            'vaso'
        ] = 0.0

        df_full[
            'iv_action'
        ] = 1

        df_full[
            'vaso_action'
        ] = 1

        df_full[
            'final_action'
        ] = 0

    df_full = (
        df_full
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n전체 ICU Action distribution"
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

    # ============================================================
    # Encoder/HMM analysis window: onset 0~96h
    # ============================================================
    df_window = df_full[df_full['hours_from_onset'].between(0, analysis_window_hours, inclusive='both')].copy()
    print(f"\nAnalysis window: onset 0-{analysis_window_hours}h | stays={df_window['stay_id'].nunique()} | rows={len(df_window)}")

    sampler = Sample(data=df_query, iterations=1000, threshold=0.05, sample_size=1000, sofa=df_window)
    df_sampled = sampler.main().rename(columns=col_mapping)
    df_sampled['charttime'] = pd.to_datetime(df_sampled['charttime'])
    if 'onset_time' in df_sampled.columns:
        df_sampled = df_sampled.drop(columns=['onset_time'])

    df_sampled = pd.merge(df_sampled, onset_df, on='stay_id', how='inner')
    df_sampled['hours_from_onset'] = (df_sampled['charttime'] - df_sampled['onset_time']).dt.total_seconds() / 3600
    df_sampled = df_sampled[df_sampled['hours_from_onset'].between(0, analysis_window_hours, inclusive='both')].copy()
    df_sampled = df_sampled.sort_values(['stay_id', 'charttime']).reset_index(drop=True)
    sampled_stay_ids = df_sampled['stay_id'].unique().copy()

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
        .groupby(
            'stay_id'
        )[
            features_col
        ]
        .ffill()
        .bfill()
        .fillna(0)
    )

    hmm_module = (
        SepsisHMM(
            n_components=4
        )
    )

    hmm_module.train(
        df_sampled,
        features_col
    )

    hmm_module.save_model()

    df_sampled[
        'hmm_state'
    ] = (
        hmm_module.predict(
            df_sampled,
            features_col
        )
    )

    next_features_cols = [f"{c}_next" for c in features_col]
    df_sampled[next_features_cols] = df_sampled.groupby('stay_id')[features_col].shift(-1)
    df_sampled['encoder_next_charttime'] = df_sampled.groupby('stay_id')['charttime'].shift(-1)
    df_sampled['encoder_transition_hours'] = (df_sampled['encoder_next_charttime'] - df_sampled['charttime']).dt.total_seconds() / 3600
    df_shifted = df_sampled.dropna(subset=next_features_cols + ['encoder_next_charttime']).copy()
    df_shifted = df_shifted[np.isclose(df_shifted['encoder_transition_hours'], interval)].copy()
    df_shifted = df_shifted.sort_values(['stay_id', 'charttime']).reset_index(drop=True)
    print(f"Encoder 4h transitions: {len(df_shifted)} | stays={df_shifted['stay_id'].nunique()}")

    encoder_stay_ids = (
        df_shifted[
            'stay_id'
        ]
        .unique()
        .copy()
    )

    rng = (
        np.random.default_rng(
            42
        )
    )

    rng.shuffle(
        encoder_stay_ids
    )

    encoder_split = int(
        len(
            encoder_stay_ids
        )
        * 0.8
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
        hmm_module
        .scaler
        .transform(
            encoder_train_df[
                features_col
            ]
            .values
        )
    )

    X_train_next = (
        hmm_module
        .scaler
        .transform(
            encoder_train_df[
                next_features_cols
            ]
            .values
        )
    )

    X_val = (
        hmm_module
        .scaler
        .transform(
            encoder_val_df[
                features_col
            ]
            .values
        )
    )

    X_val_next = (
        hmm_module
        .scaler
        .transform(
            encoder_val_df[
                next_features_cols
            ]
            .values
        )
    )

    train_dataset = (
        TensorDataset(
            torch.FloatTensor(
                X_train
            ),
            torch.FloatTensor(
                X_train_next
            ),
            torch.LongTensor(
                encoder_train_df[
                    'hmm_state'
                ]
                .values
            )
        )
    )

    val_dataset = (
        TensorDataset(
            torch.FloatTensor(
                X_val
            ),
            torch.FloatTensor(
                X_val_next
            ),
            torch.LongTensor(
                encoder_val_df[
                    'hmm_state'
                ]
                .values
            )
        )
    )

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=
                batch_size,
            shuffle=True,
            drop_last=False
        )
    )

    val_dataloader = (
        DataLoader(
            val_dataset,
            batch_size=
                batch_size,
            shuffle=False,
            drop_last=False
        )
    )

    encoder_module = (
        SepsisEncoder(
            input_dim=
                len(
                    features_col
                ),
            latent_dim=7,
            device=
                str(device)
        )
    )

    encoder_module.train(
        train_loader=
            train_dataloader,
        val_loader=
            val_dataloader,
        epochs=
            encoder_epochs
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

    # ============================================================
    # RL window: same onset 0~96h horizon as Encoder/HMM
    # ============================================================
    df_rl = df_full[(~df_full['stay_id'].isin(sampled_stay_ids)) & df_full['hours_from_onset'].between(0, analysis_window_hours, inclusive='both')].copy()
    df_rl = df_rl.sort_values(['stay_id', 'charttime']).reset_index(drop=True)
    print(f"RL analysis window: onset 0-{analysis_window_hours}h | stays={df_rl['stay_id'].nunique()} | rows={len(df_rl)}")

    overlap = (
        set(
            sampled_stay_ids
        )
        &
        set(
            df_rl[
                'stay_id'
            ]
            .unique()
        )
    )

    print(
        "\nEncoder/RL overlap:",
        len(overlap)
    )

    if len(overlap) > 0:

        raise ValueError(
            "Encoder/RL overlap"
        )

    df_rl[
        features_col
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
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
        .groupby(
            'stay_id'
        )[
            features_col
        ]
        .shift(-1)
    )

    df_rl[
        'sofa_score_next'
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
            'sofa_score'
        ]
        .shift(-1)
    )

    df_rl[
        'next_charttime'
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
            'charttime'
        ]
        .shift(-1)
    )

    df_rl[
        'action_to_next'
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
            'final_action'
        ]
        .shift(-1)
    )

    df_rl[
        'iv_fluid_to_next'
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
            'iv_fluid'
        ]
        .shift(-1)
    )

    df_rl[
        'vaso_to_next'
    ] = (
        df_rl
        .groupby(
            'stay_id'
        )[
            'vaso'
        ]
        .shift(-1)
    )

    df_rl[
        'transition_hours'
    ] = (
        (
            df_rl[
                'next_charttime'
            ]
            -
            df_rl[
                'charttime'
            ]
        )
        .dt
        .total_seconds()
        / 3600
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

    df_rl_shifted = (
        df_rl_shifted[
            np.isclose(
                df_rl_shifted[
                    'transition_hours'
                ],
                interval
            )
        ]
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

    df_rl_shifted = (
        df_rl_shifted
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df_rl_shifted[
        'is_last'
    ] = (
        df_rl_shifted
        .groupby(
            'stay_id'
        )
        .cumcount(
            ascending=False
        )
        == 0
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

    train_end = int(
        n_rl * 0.70
    )

    val_end = int(
        n_rl * 0.85
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

    rl_train_dataloader = (
        DataLoader(
            rl_train_dataset,
            batch_size=
                batch_size,
            shuffle=True,
            drop_last=False
        )
    )

    rl_val_dataloader = (
        DataLoader(
            rl_val_dataset,
            batch_size=
                batch_size,
            shuffle=False,
            drop_last=False
        )
    )

    rl_test_dataloader = (
        DataLoader(
            rl_test_dataset,
            batch_size=
                batch_size,
            shuffle=False,
            drop_last=False
        )
    )

    q_net = (
        LowLevelQNetwork(
            latent_dim=7,
            action_dim=25
        )
        .to(device)
    )

    p_net = (
        LowLevelQNetwork(
            latent_dim=7,
            action_dim=25
        )
        .to(device)
    )

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

    best_low_val = float(
        'inf'
    )

    for epoch in range(
        low_level_epochs
    ):

        q_net.train()
        p_net.train()

        total_q = 0.0
        total_p = 0.0

        total_q_td = 0.0
        total_p_td = 0.0

        total_q_cql = 0.0
        total_p_cql = 0.0

        batches = 0

        for (
            batch_X,
            batch_X_next,
            batch_act,
            batch_sofa,
            batch_sofa_next,
            batch_surv,
            batch_last
        ) in rl_train_dataloader:

            batch_X = (
                batch_X.to(
                    device
                )
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

                (
                    reward_q,
                    reward_p
                ) = make_rewards(
                    batch_sofa,
                    batch_sofa_next,
                    batch_surv,
                    batch_last,
                    lambda_pt
                )

            q_all = (
                q_net(
                    z_curr
                )
            )

            q_data = (
                q_all
                .gather(
                    1,
                    batch_act
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            with torch.no_grad():

                next_q_action = (
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
                        next_q_action
                    )
                    .squeeze(1)
                )

                q_target = (
                    reward_q
                    + gamma
                    * next_q
                    * (
                        ~batch_last
                    ).float()
                )

            q_td = (
                F.smooth_l1_loss(
                    q_data,
                    q_target
                )
            )

            q_cql = (
                cql_loss(
                    q_all,
                    q_data
                )
            )

            q_loss = (
                q_td
                + cql_weight
                * q_cql
            )

            opt_q.zero_grad()

            q_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                q_net.parameters(),
                1.0
            )

            opt_q.step()

            p_all = (
                p_net(
                    z_curr
                )
            )

            p_data = (
                p_all
                .gather(
                    1,
                    batch_act
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            with torch.no_grad():

                next_p_action = (
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
                        next_p_action
                    )
                    .squeeze(1)
                )

                p_target = (
                    reward_p
                    + gamma
                    * next_p
                    * (
                        ~batch_last
                    ).float()
                )

            p_td = (
                F.smooth_l1_loss(
                    p_data,
                    p_target
                )
            )

            p_cql = (
                cql_loss(
                    p_all,
                    p_data
                )
            )

            p_loss = (
                p_td
                + cql_weight
                * p_cql
            )

            opt_p.zero_grad()

            p_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                p_net.parameters(),
                1.0
            )

            opt_p.step()

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

            total_q += (
                q_loss.item()
            )

            total_p += (
                p_loss.item()
            )

            total_q_td += (
                q_td.item()
            )

            total_p_td += (
                p_td.item()
            )

            total_q_cql += (
                q_cql.item()
            )

            total_p_cql += (
                p_cql.item()
            )

            batches += 1

        val_low = (
            evaluate_low_level(
                rl_val_dataloader,
                q_net,
                p_net,
                target_q,
                target_p,
                encoder_module,
                device,
                gamma,
                lambda_pt,
                cql_weight
            )
        )

        print(
            f"\nLow-Level Epoch "
            f"{epoch + 1:03d}"
        )

        print(
            f"Train Q: "
            f"{total_q / batches:.4f}"
            f" | P: "
            f"{total_p / batches:.4f}"
        )

        print(
            f"Validation Q: "
            f"{val_low['q_loss']:.4f}"
            f" | P: "
            f"{val_low['p_loss']:.4f}"
        )

        print(
            f"Validation Q Agreement: "
            f"{val_low['q_agreement']:.4f}"
            f" | Prospect: "
            f"{val_low['p_agreement']:.4f}"
        )

        print(
            f"Validation Q Nonzero: "
            f"{val_low['q_nonzero_agreement']:.4f}"
            f" | Prospect Nonzero: "
            f"{val_low['p_nonzero_agreement']:.4f}"
        )

        if (
            val_low[
                'total_loss'
            ]
            <
            best_low_val
        ):

            best_low_val = (
                val_low[
                    'total_loss'
                ]
            )

            torch.save(
                {
                    'q_net':
                        q_net.state_dict(),

                    'p_net':
                        p_net.state_dict(),

                    'target_q':
                        target_q.state_dict(),

                    'target_p':
                        target_p.state_dict(),

                    'val_loss':
                        best_low_val
                },
                'low_level_best.pth'
            )

    low_checkpoint = torch.load(
        'low_level_best.pth',
        map_location=device
    )

    q_net.load_state_dict(
        low_checkpoint[
            'q_net'
        ]
    )

    p_net.load_state_dict(
        low_checkpoint[
            'p_net'
        ]
    )

    target_q.load_state_dict(
        low_checkpoint[
            'target_q'
        ]
    )

    target_p.load_state_dict(
        low_checkpoint[
            'target_p'
        ]
    )

    q_net.eval()
    p_net.eval()

    for param in (
        q_net.parameters()
    ):
        param.requires_grad = False

    for param in (
        p_net.parameters()
    ):
        param.requires_grad = False

    (
        g_mean,
        g_std
    ) = compute_uncertainty_stats(
        rl_train_dataloader,
        encoder_module,
        device
    )

    (
        margin_mean,
        margin_std
    ) = compute_option_margin_stats(
        rl_train_dataloader,
        encoder_module,
        q_net,
        p_net,
        device
    )

    print(
        "\nUncertainty mean:",
        g_mean
    )

    print(
        "Uncertainty std:",
        g_std
    )

    print(
        "Option margin mean:",
        margin_mean
    )

    print(
        "Option margin std:",
        margin_std
    )

    high_net = (
        HighLevelQNetwork(
            latent_dim=7,
            num_options=2
        )
        .to(device)
    )

    target_high = (
        copy.deepcopy(
            high_net
        )
        .to(device)
    )

    target_high.eval()

    for param in (
        target_high.parameters()
    ):
        param.requires_grad = False

    opt_high = (
        optim.Adam(
            high_net.parameters(),
            lr=3e-4
        )
    )

    best_high_val = float(
        'inf'
    )

    for epoch in range(
        high_level_epochs
    ):

        high_net.train()

        total_loss = 0.0
        total_td = 0.0
        total_cql = 0.0

        behavior_q_count = 0
        behavior_p_count = 0

        pred_q_count = 0
        pred_p_count = 0

        q_uncertainty_sum = 0.0
        p_uncertainty_sum = 0.0

        total_samples = 0
        batches = 0

        for (
            batch_X,
            batch_X_next,
            batch_act,
            batch_sofa,
            batch_sofa_next,
            batch_surv,
            batch_last
        ) in rl_train_dataloader:

            batch_X = (
                batch_X.to(
                    device
                )
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

                g_curr = get_uncertainty(
                    encoder_module,
                    z_curr
                )

                g_next = get_uncertainty(
                    encoder_module,
                    z_next
                )

                g_curr = scale_uncertainty(
                    g_curr,
                    g_mean,
                    g_std
                )

                g_next = scale_uncertainty(
                    g_next,
                    g_mean,
                    g_std
                )

                q_values = (
                    q_net(
                        z_curr
                    )
                )

                p_values = (
                    p_net(
                        z_curr
                    )
                )

                (
                    behavior_option,
                    _,
                    _,
                    _
                ) = infer_high_level_option(
                    q_values,
                    p_values,
                    batch_act,
                    g_curr,
                    margin_mean,
                    margin_std,
                    behavior_option_weight,
                    uncertainty_option_weight
                )

                (
                    reward_q,
                    _
                ) = make_rewards(
                    batch_sofa,
                    batch_sofa_next,
                    batch_surv,
                    batch_last,
                    lambda_pt
                )

                (
                    reward_high,
                    _
                ) = make_high_level_reward(
                    reward_q,
                    behavior_option,
                    g_curr,
                    uncertainty_reward_weight
                )

            high_q_all = (
                high_net(
                    z_curr,
                    g_curr
                )
            )

            high_q_data = (
                high_q_all
                .gather(
                    1,
                    behavior_option
                    .unsqueeze(1)
                )
                .squeeze(1)
            )

            with torch.no_grad():

                next_option = (
                    high_net(
                        z_next,
                        g_next
                    )
                    .argmax(
                        dim=1,
                        keepdim=True
                    )
                )

                next_high_q = (
                    target_high(
                        z_next,
                        g_next
                    )
                    .gather(
                        1,
                        next_option
                    )
                    .squeeze(1)
                )

                high_target = (
                    reward_high
                    + gamma
                    * next_high_q
                    * (
                        ~batch_last
                    ).float()
                )

            high_td = (
                F.smooth_l1_loss(
                    high_q_data,
                    high_target
                )
            )

            high_cql = (
                cql_loss(
                    high_q_all,
                    high_q_data
                )
            )

            high_loss = (
                high_td
                + high_cql_weight
                * high_cql
            )

            opt_high.zero_grad()

            high_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                high_net.parameters(),
                1.0
            )

            opt_high.step()

            soft_update(
                target_high,
                high_net,
                tau
            )

            with torch.no_grad():

                pred_option = (
                    high_net(
                        z_curr,
                        g_curr
                    )
                    .argmax(
                        dim=1
                    )
                )

            behavior_q_count += (
                behavior_option
                == 0
            ).sum().item()

            behavior_p_count += (
                behavior_option
                == 1
            ).sum().item()

            pred_q_mask = (
                pred_option == 0
            )

            pred_p_mask = (
                pred_option == 1
            )

            pred_q_count += (
                pred_q_mask
                .sum()
                .item()
            )

            pred_p_count += (
                pred_p_mask
                .sum()
                .item()
            )

            if pred_q_mask.any():

                q_uncertainty_sum += (
                    g_curr[
                        pred_q_mask
                    ]
                    .sum()
                    .item()
                )

            if pred_p_mask.any():

                p_uncertainty_sum += (
                    g_curr[
                        pred_p_mask
                    ]
                    .sum()
                    .item()
                )

            total_samples += (
                batch_act.size(0)
            )

            total_loss += (
                high_loss.item()
            )

            total_td += (
                high_td.item()
            )

            total_cql += (
                high_cql.item()
            )

            batches += 1

        val_high = (
            evaluate_high_level(
                rl_val_dataloader,
                q_net,
                p_net,
                high_net,
                target_high,
                encoder_module,
                device,
                gamma,
                lambda_pt,
                high_cql_weight,
                g_mean,
                g_std,
                margin_mean,
                margin_std,
                behavior_option_weight,
                uncertainty_option_weight,
                uncertainty_reward_weight
            )
        )

        train_q_uncertainty = (
            q_uncertainty_sum
            / pred_q_count
            if pred_q_count > 0
            else np.nan
        )

        train_p_uncertainty = (
            p_uncertainty_sum
            / pred_p_count
            if pred_p_count > 0
            else np.nan
        )

        print(
            f"\nHigh-Level Epoch "
            f"{epoch + 1:03d}"
        )

        print(
            f"Train Loss: "
            f"{total_loss / batches:.4f}"
            f" | TD: "
            f"{total_td / batches:.4f}"
            f" | CQL: "
            f"{total_cql / batches:.4f}"
        )

        print(
            f"Behavior Option Q/P: "
            f"{behavior_q_count / total_samples:.4f}"
            f" / "
            f"{behavior_p_count / total_samples:.4f}"
        )

        print(
            f"Predicted Option Q/P: "
            f"{pred_q_count / total_samples:.4f}"
            f" / "
            f"{pred_p_count / total_samples:.4f}"
        )

        print(
            f"Train Mean Uncertainty "
            f"Q/Prospect: "
            f"{train_q_uncertainty:.4f}"
            f" / "
            f"{train_p_uncertainty:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_high['loss']:.4f}"
            f" | Option Agreement: "
            f"{val_high['option_agreement']:.4f}"
        )

        print(
            f"Validation Hierarchy: "
            f"{val_high['hierarchy_agreement']:.4f}"
            f" | Nonzero: "
            f"{val_high['hierarchy_nonzero_agreement']:.4f}"
        )

        print(
            f"Validation Prospect Ratio: "
            f"{val_high['prospect_ratio']:.4f}"
            f" | Behavior Prospect Ratio: "
            f"{val_high['behavior_prospect_ratio']:.4f}"
        )

        print(
            f"Validation Uncertainty "
            f"Q/Prospect: "
            f"{val_high['pred_q_uncertainty']:.4f}"
            f" / "
            f"{val_high['pred_prospect_uncertainty']:.4f}"
        )

        print(
            f"Validation Uncertainty Gap: "
            f"{val_high['uncertainty_gap']:.4f}"
            f" | Alignment: "
            f"{val_high['mean_alignment']:.4f}"
        )

        if (
            val_high[
                'loss'
            ]
            <
            best_high_val
        ):

            best_high_val = (
                val_high[
                    'loss'
                ]
            )

            torch.save(
                {
                    'q_net':
                        q_net.state_dict(),

                    'p_net':
                        p_net.state_dict(),

                    'high_net':
                        high_net.state_dict(),

                    'target_high':
                        target_high.state_dict(),

                    'g_mean':
                        g_mean,

                    'g_std':
                        g_std,

                    'margin_mean':
                        margin_mean,

                    'margin_std':
                        margin_std,

                    'behavior_option_weight':
                        behavior_option_weight,

                    'uncertainty_option_weight':
                        uncertainty_option_weight,

                    'uncertainty_reward_weight':
                        uncertainty_reward_weight,

                    'gamma':
                        gamma,

                    'lambda_pt':
                        lambda_pt,

                    'cql_weight':
                        cql_weight,

                    'high_cql_weight':
                        high_cql_weight,

                    'val_loss':
                        best_high_val,

                    'epoch':
                        epoch + 1
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

    high_net.load_state_dict(
        checkpoint[
            'high_net'
        ]
    )

    target_high.load_state_dict(
        checkpoint[
            'target_high'
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

    margin_mean = (
        checkpoint[
            'margin_mean'
        ]
    )

    margin_std = (
        checkpoint[
            'margin_std'
        ]
    )

    test_low = (
        evaluate_low_level(
            rl_test_dataloader,
            q_net,
            p_net,
            target_q,
            target_p,
            encoder_module,
            device,
            gamma,
            lambda_pt,
            cql_weight
        )
    )

    test_high = (
        evaluate_high_level(
            rl_test_dataloader,
            q_net,
            p_net,
            high_net,
            target_high,
            encoder_module,
            device,
            gamma,
            lambda_pt,
            high_cql_weight,
            g_mean,
            g_std,
            margin_mean,
            margin_std,
            behavior_option_weight,
            uncertainty_option_weight,
            uncertainty_reward_weight
        )
    )

    print(
        "\nBest High-Level Epoch:",
        checkpoint[
            'epoch'
        ]
    )

    print(
        "Best High-Level Validation Loss:",
        checkpoint[
            'val_loss'
        ]
    )

    print(
        "\nTest Q Agreement:",
        test_low[
            'q_agreement'
        ]
    )

    print(
        "Test Prospect Agreement:",
        test_low[
            'p_agreement'
        ]
    )

    print(
        "Test Q Nonzero Agreement:",
        test_low[
            'q_nonzero_agreement'
        ]
    )

    print(
        "Test Prospect Nonzero Agreement:",
        test_low[
            'p_nonzero_agreement'
        ]
    )

    print(
        "\nTest High-Level Loss:",
        test_high[
            'loss'
        ]
    )

    print(
        "Test Option Agreement:",
        test_high[
            'option_agreement'
        ]
    )

    print(
        "Test Hierarchy Agreement:",
        test_high[
            'hierarchy_agreement'
        ]
    )

    print(
        "Test Hierarchy Nonzero Agreement:",
        test_high[
            'hierarchy_nonzero_agreement'
        ]
    )

    print(
        "Test Prospect Selection Ratio:",
        test_high[
            'prospect_ratio'
        ]
    )

    print(
        "Test Behavior Prospect Ratio:",
        test_high[
            'behavior_prospect_ratio'
        ]
    )

    print(
        "Test Mean Uncertainty Q Option:",
        test_high[
            'pred_q_uncertainty'
        ]
    )

    print(
        "Test Mean Uncertainty Prospect Option:",
        test_high[
            'pred_prospect_uncertainty'
        ]
    )

    print(
        "Test Uncertainty Gap:",
        test_high[
            'uncertainty_gap'
        ]
    )

    print(
        "Test Uncertainty Alignment:",
        test_high[
            'mean_alignment'
        ]
    )

    print(
        "Test Predicted Action0 Ratio:",
        test_high[
            'predicted_action0_ratio'
        ]
    )
    result_visualizer = visualizer(
    output_dir=
        PROJECT_ROOT
        / "results"
    )

    print(
        "\n=============================="
    )
    print(
        "RESULT EXTRACTION"
    )
    print(
        "=============================="
    )

    policy_df = (
        result_visualizer
        .extract_policy_results(
            df_eval=
                df_rl_test,
            scaler=
                hmm_module.scaler,
            features_col=
                features_col,
            encoder_module=
                encoder_module,
            q_net=
                q_net,
            p_net=
                p_net,
            high_net=
                high_net,
            g_mean=
                g_mean,
            g_std=
                g_std
        )
    )

    print(
        "\nPolicy prediction results saved"
    )

    test_summary = pd.DataFrame(
        [
            {
                'Q Agreement':
                    test_low.get(
                        'q_agreement',
                        np.nan
                    ),

                'Prospect Agreement':
                    test_low.get(
                        'p_agreement',
                        np.nan
                    ),

                'Q Nonzero Agreement':
                    test_low.get(
                        'q_nonzero_agreement',
                        np.nan
                    ),

                'Prospect Nonzero Agreement':
                    test_low.get(
                        'p_nonzero_agreement',
                        np.nan
                    ),

                'HRL Agreement':
                    test_high.get(
                        'hierarchy_agreement',
                        np.nan
                    ),

                'HRL Nonzero Agreement':
                    test_high.get(
                        'hierarchy_nonzero_agreement',
                        np.nan
                    ),

                'Option Agreement':
                    test_high.get(
                        'option_agreement',
                        np.nan
                    ),

                'Prospect Selection Ratio':
                    test_high.get(
                        'prospect_ratio',
                        np.nan
                    ),

                'Behavior Prospect Ratio':
                    test_high.get(
                        'behavior_prospect_ratio',
                        np.nan
                    ),

                'Mean Uncertainty Q':
                    test_high.get(
                        'mean_uncertainty_q',
                        np.nan
                    ),

                'Mean Uncertainty Prospect':
                    test_high.get(
                        'mean_uncertainty_p',
                        np.nan
                    ),

                'Uncertainty Gap':
                    test_high.get(
                        'uncertainty_gap',
                        np.nan
                    ),

                'Predicted Action0 Ratio':
                    test_high.get(
                        'predicted_action0_ratio',
                        np.nan
                    )
            }
        ]
    )

    test_summary.to_csv(
        result_visualizer.eval_dir
        / "test_summary.csv",
        index=False
    )

    print(
        "\nTest Summary"
    )

    print(
        test_summary.T
    )

    clinician_results = (
        result_visualizer
        .evaluate_vs_clinicians(
            policy_df
        )
    )

    print(
        "\n=============================="
    )
    print(
        "VS CLINICIANS"
    )
    print(
        "=============================="
    )

    print(
        clinician_results
    )

    uncertainty_option_summary = (
        policy_df
        .assign(
            Option=np.where(
                policy_df[
                    'high_option'
                ]
                == 0,
                'Q',
                'Prospect'
            )
        )
        .groupby(
            'Option'
        )[
            'uncertainty'
        ]
        .agg(
            [
                'count',
                'mean',
                'std',
                'median'
            ]
        )
        .reset_index()
    )

    uncertainty_option_summary.to_csv(
        result_visualizer.eval_dir
        / "uncertainty_by_option.csv",
        index=False
    )

    print(
        "\n=============================="
    )
    print(
        "UNCERTAINTY BY OPTION"
    )
    print(
        "=============================="
    )

    print(
        uncertainty_option_summary
    )

    uncertainty_df = (
        policy_df[
            [
                'stay_id',
                'charttime',
                'uncertainty',
                'high_option'
            ]
        ]
        .copy()
    )

    uncertainty_df[
        'uncertainty_decile'
    ] = (
        pd.qcut(
            uncertainty_df[
                'uncertainty'
            ],
            q=10,
            labels=False,
            duplicates='drop'
        )
        + 1
    )

    uncertainty_decile_results = (
        uncertainty_df
        .groupby(
            'uncertainty_decile',
            as_index=False
        )
        .agg(
            Mean_Uncertainty=(
                'uncertainty',
                'mean'
            ),
            Prospect_Selection_Rate=(
                'high_option',
                lambda x:
                    (
                        x == 1
                    ).mean()
            ),
            N=(
                'high_option',
                'size'
            )
        )
    )

    uncertainty_decile_results.to_csv(
        result_visualizer.eval_dir
        / "uncertainty_decile_prospect_selection.csv",
        index=False
    )

    print(
        "\n=============================="
    )
    print(
        "UNCERTAINTY DECILE"
    )
    print(
        "=============================="
    )

    print(
        uncertainty_decile_results
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        uncertainty_decile_results[
            'Mean_Uncertainty'
        ],
        uncertainty_decile_results[
            'Prospect_Selection_Rate'
        ],
        marker='o',
        linewidth=2
    )

    ax.set_xlabel(
        'SDE Diffusion Uncertainty'
    )

    ax.set_ylabel(
        'Prospect Selection Ratio'
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        'SDE Uncertainty vs Prospect Selection'
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        result_visualizer.eval_dir
        / "uncertainty_vs_prospect_selection.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    option_counts = (
        policy_df[
            'high_option'
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    option_distribution = (
        pd.DataFrame(
            {
                'Option':
                    [
                        'Q',
                        'Prospect'
                    ],

                'Selection Ratio':
                    [
                        option_counts.get(
                            0,
                            0.0
                        ),
                        option_counts.get(
                            1,
                            0.0
                        )
                    ]
            }
        )
    )

    option_distribution.to_csv(
        result_visualizer.eval_dir
        / "high_level_option_distribution.csv",
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    bars = ax.bar(
        option_distribution[
            'Option'
        ],
        option_distribution[
            'Selection Ratio'
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        'Selection Ratio'
    )

    ax.set_title(
        'High-Level Option Distribution'
    )

    for bar, value in zip(
        bars,
        option_distribution[
            'Selection Ratio'
        ]
    ):
        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha='center',
            va='bottom'
        )

    fig.tight_layout()

    fig.savefig(
        result_visualizer.eval_dir
        / "high_level_option_distribution.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    mortality_q_results = (
        result_visualizer
        .mortality_vs_q_value(
            policy_df=
                policy_df,
            n_bins=10
        )
    )

    print(
        "\n=============================="
    )
    print(
        "MORTALITY VS Q VALUE"
    )
    print(
        "=============================="
    )

    print(
        mortality_q_results
    )

    action_mortality_results = (
        result_visualizer
        .action_level_vs_mortality(
            policy_df
        )
    )

    print(
        "\n=============================="
    )
    print(
        "CLINICIAN ACTION VS MORTALITY"
    )
    print(
        "=============================="
    )

    print(
        action_mortality_results[
            'Clinician'
        ]
    )

    print(
        "\n=============================="
    )
    print(
        "HRL ACTION VS MORTALITY"
    )
    print(
        "=============================="
    )

    print(
        action_mortality_results[
            'HRL'
        ]
    )

    wis_results = (
        result_visualizer
        .evaluate_wis(
            df_train=
                df_rl_train,
            df_test=
                df_rl_test,
            scaler=
                hmm_module.scaler,
            features_col=
                features_col,
            encoder_module=
                encoder_module,
            q_net=
                q_net,
            p_net=
                p_net,
            high_net=
                high_net,
            g_mean=
                g_mean,
            g_std=
                g_std,
            gamma=
                gamma,
            temperature=
                1.0,
            behavior_epochs=
                5
        )
    )

    print(
        "\n=============================="
    )
    print(
        "WIS"
    )
    print(
        "=============================="
    )

    print(
        wis_results
    )

    if (
        wis_results[
            'ESS'
        ].min()
        < 10
    ):
        print(
            "\nWARNING: "
            "WIS ESS is very low. "
            "Do not use this WIS value "
            "as a primary policy evaluation result."
        )

    policy_value_results = (
        result_visualizer
        .save_policy_value(
            wis_results
        )
    )

    print(
        "\n=============================="
    )
    print(
        "POLICY VALUE"
    )
    print(
        "=============================="
    )

    print(
        policy_value_results
    )

    q_prospect_results = (
        result_visualizer
        .compare_only_q_prospect(
            policy_df=
                policy_df,
            wis_summary=
                wis_results
        )
    )

    print(
        "\n=============================="
    )
    print(
        "ONLY Q VS ONLY PROSPECT"
    )
    print(
        "=============================="
    )

    print(
        q_prospect_results
    )

    hyperparameter_results = (
        result_visualizer
        .record_hyperparameter_result(
            params={
                'gamma':
                    gamma,

                'lambda_pt':
                    lambda_pt,

                'cql_weight':
                    cql_weight,

                'high_cql_weight':
                    high_cql_weight,

                'low_level_epochs':
                    low_level_epochs,

                'high_level_epochs':
                    high_level_epochs
            },
            test_high=
                test_high,
            test_low=
                test_low,
            wis_summary=
                wis_results
        )
    )

    print(
        "\n=============================="
    )
    print(
        "HYPERPARAMETER RESULTS"
    )
    print(
        "=============================="
    )

    print(
        hyperparameter_results.tail()
    )

    latent_tsne_results = (
        result_visualizer
        .visualize_latent_tsne_full(
            sepsis_encoder=
                encoder_module,
            data_loader=
                val_dataloader
        )
    )

    print(
        "\nLatent t-SNE saved"
    )

    fg_norm_results = (
        result_visualizer
        .extract_and_visualize_fg_norms_by_stage(
            encoder_module=
                encoder_module,
            data_loader=
                val_dataloader
        )
    )

    print(
        "\nSDE f/g norm results saved"
    )

    sde_trajectory_results = (
        result_visualizer
        .visualize_sde_trajectory_pca(
            encoder_module=
                encoder_module,
            data_loader=
                val_dataloader,
            num_samples=5,
            time_steps=50
        )
    )

    print(
        "\nSDE trajectory results saved"
    )

    df_hmm_result = (
        df_sampled.copy()
    )

    if (
        'sofa_score'
        not in df_hmm_result.columns
    ):

        sofa_result = (
            df_window[
                [
                    'stay_id',
                    'charttime',
                    'sofa_score'
                ]
            ]
            .drop_duplicates(
                subset=[
                    'stay_id',
                    'charttime'
                ]
            )
            .copy()
        )

        df_hmm_result[
            'charttime'
        ] = pd.to_datetime(
            df_hmm_result[
                'charttime'
            ]
        )

        sofa_result[
            'charttime'
        ] = pd.to_datetime(
            sofa_result[
                'charttime'
            ]
        )

        df_hmm_result = pd.merge(
            df_hmm_result,
            sofa_result,
            on=[
                'stay_id',
                'charttime'
            ],
            how='left'
        )

    hmm_results = (
        result_visualizer
        .save_hmm_results(
            df_hmm=
                df_hmm_result,
            hmm_module=
                hmm_module
        )
    )

    print(
        "\n=============================="
    )
    print(
        "HMM RESULTS"
    )
    print(
        "=============================="
    )

    print(
        hmm_results
    )

    final_result_summary = pd.DataFrame(
        [
            {
                'HRL Agreement':
                    test_high.get(
                        'hierarchy_agreement',
                        np.nan
                    ),

                'HRL Nonzero Agreement':
                    test_high.get(
                        'hierarchy_nonzero_agreement',
                        np.nan
                    ),

                'Q Agreement':
                    test_low.get(
                        'q_agreement',
                        np.nan
                    ),

                'Prospect Agreement':
                    test_low.get(
                        'p_agreement',
                        np.nan
                    ),

                'Option Agreement':
                    test_high.get(
                        'option_agreement',
                        np.nan
                    ),

                'Prospect Ratio':
                    (
                        policy_df[
                            'high_option'
                        ]
                        == 1
                    ).mean(),

                'Mean Q Option Uncertainty':
                    policy_df.loc[
                        policy_df[
                            'high_option'
                        ]
                        == 0,
                        'uncertainty'
                    ].mean(),

                'Mean Prospect Option Uncertainty':
                    policy_df.loc[
                        policy_df[
                            'high_option'
                        ]
                        == 1,
                        'uncertainty'
                    ].mean(),

                'Uncertainty Gap':
                    (
                        policy_df.loc[
                            policy_df[
                                'high_option'
                            ]
                            == 1,
                            'uncertainty'
                        ].mean()
                        -
                        policy_df.loc[
                            policy_df[
                                'high_option'
                            ]
                            == 0,
                            'uncertainty'
                        ].mean()
                    ),

                'HRL WIS':
                    wis_results.loc[
                        wis_results[
                            'Policy'
                        ]
                        == 'HRL',
                        'WIS'
                    ].iloc[0],

                'HRL WIS ESS':
                    wis_results.loc[
                        wis_results[
                            'Policy'
                        ]
                        == 'HRL',
                        'ESS'
                    ].iloc[0]
            }
        ]
    )

    final_result_summary.to_csv(
        result_visualizer.output_dir
        / "final_result_summary.csv",
        index=False
    )

    print(
        "\n=============================="
    )
    print(
        "FINAL RESULT SUMMARY"
    )
    print(
        "=============================="
    )

    print(
        final_result_summary.T
    )

    print(
        "\nAll results saved to:"
    )

    print(
        result_visualizer.output_dir
    )
    # ============================================================
    # VALIDATION EXPERIMENTS
    # ============================================================

    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt

    print(
        "\n"
        "============================================================"
    )
    print(
        "VALIDATION EXPERIMENTS"
    )
    print(
        "============================================================"
    )

    validation_dir = (
        PROJECT_ROOT
        / "results"
        / "validation"
    )

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 0. policy_df가 없는 경우 생성
    # ------------------------------------------------------------

    if (
        'policy_df'
        not in globals()
    ):
        result_visualizer = visualizer(
            output_dir=
                PROJECT_ROOT
                / "results"
        )

        policy_df = (
            result_visualizer
            .extract_policy_results(
                df_eval=
                    df_rl_test,
                scaler=
                    hmm_module.scaler,
                features_col=
                    features_col,
                encoder_module=
                    encoder_module,
                q_net=
                    q_net,
                p_net=
                    p_net,
                high_net=
                    high_net,
                g_mean=
                    g_mean,
                g_std=
                    g_std
            )
        )

    device = torch.device(
        str(
            encoder_module.device
        )
    )

    q_net.eval()
    p_net.eval()
    high_net.eval()

    encoder_module.encoder.eval()
    encoder_module.sde.eval()

    # ============================================================
    # Helper functions
    # ============================================================

    def validation_get_latent(
        x
    ):
        with torch.no_grad():

            mu, _ = (
                encoder_module
                .encoder(
                    x
                )
            )

        return mu


    def validation_get_uncertainty(
        z
    ):
        with torch.no_grad():

            t_zero = (
                torch.zeros_like(
                    z[:, :1]
                )
            )

            ty = torch.cat(
                [
                    t_zero,
                    z
                ],
                dim=1
            )

            g_val = (
                encoder_module
                .sde
                .g_net(
                    ty
                )
                + 1e-3
            )

            g_raw = (
                torch.max(
                    g_val,
                    dim=1
                )
                .values
            )

            g_scaled = (
                torch.sigmoid(
                    (
                        g_raw
                        - g_mean
                    )
                    /
                    (
                        g_std
                        + 1e-6
                    )
                )
            )

        return (
            g_raw,
            g_scaled
        )


    def validation_encode_dataframe(
        df,
        batch_size=4096
    ):
        X_np = (
            hmm_module
            .scaler
            .transform(
                df[
                    features_col
                ]
                .values
            )
        )

        z_list = []
        g_raw_list = []
        g_scaled_list = []

        with torch.no_grad():

            for start in range(
                0,
                len(df),
                batch_size
            ):

                end = min(
                    start
                    + batch_size,
                    len(df)
                )

                x = (
                    torch
                    .FloatTensor(
                        X_np[
                            start:end
                        ]
                    )
                    .to(
                        device
                    )
                )

                z = (
                    validation_get_latent(
                        x
                    )
                )

                (
                    g_raw,
                    g_scaled
                ) = (
                    validation_get_uncertainty(
                        z
                    )
                )

                z_list.append(
                    z.cpu().numpy()
                )

                g_raw_list.append(
                    g_raw.cpu().numpy()
                )

                g_scaled_list.append(
                    g_scaled
                    .cpu()
                    .numpy()
                )

        return (
            X_np,
            np.concatenate(
                z_list,
                axis=0
            ),
            np.concatenate(
                g_raw_list
            ),
            np.concatenate(
                g_scaled_list
            )
        )


    def safe_spearman(
        x,
        y
    ):
        temp = pd.DataFrame(
            {
                'x':
                    np.asarray(
                        x
                    ),

                'y':
                    np.asarray(
                        y
                    )
            }
        )

        temp = (
            temp
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        if (
            len(
                temp
            )
            < 3
        ):
            return np.nan

        return (
            temp[
                'x'
            ]
            .corr(
                temp[
                    'y'
                ],
                method='spearman'
            )
        )


    # ============================================================
    # 1. Diffusion uncertainty vs actual transition magnitude
    # ============================================================

    print(
        "\n[1] "
        "Diffusion uncertainty "
        "vs actual transition magnitude"
    )

    df_transition_validation = (
        df_rl_test
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    X_curr_np = (
        hmm_module
        .scaler
        .transform(
            df_transition_validation[
                features_col
            ]
            .values
        )
    )

    X_next_np = (
        hmm_module
        .scaler
        .transform(
            df_transition_validation[
                next_features_cols
            ]
            .values
        )
    )

    uncertainty_list = []
    latent_change_list = []
    feature_change_list = []

    batch_size_validation = 4096

    with torch.no_grad():

        for start in range(
            0,
            len(
                df_transition_validation
            ),
            batch_size_validation
        ):

            end = min(
                start
                + batch_size_validation,
                len(
                    df_transition_validation
                )
            )

            x_curr = (
                torch
                .FloatTensor(
                    X_curr_np[
                        start:end
                    ]
                )
                .to(
                    device
                )
            )

            x_next = (
                torch
                .FloatTensor(
                    X_next_np[
                        start:end
                    ]
                )
                .to(
                    device
                )
            )

            z_curr = (
                validation_get_latent(
                    x_curr
                )
            )

            z_next = (
                validation_get_latent(
                    x_next
                )
            )

            (
                _,
                g_scaled
            ) = (
                validation_get_uncertainty(
                    z_curr
                )
            )

            latent_change = (
                torch.norm(
                    z_next
                    - z_curr,
                    p=2,
                    dim=1
                )
            )

            feature_change = (
                torch.norm(
                    x_next
                    - x_curr,
                    p=2,
                    dim=1
                )
            )

            uncertainty_list.append(
                g_scaled
                .cpu()
                .numpy()
            )

            latent_change_list.append(
                latent_change
                .cpu()
                .numpy()
            )

            feature_change_list.append(
                feature_change
                .cpu()
                .numpy()
            )

    transition_uncertainty = (
        np.concatenate(
            uncertainty_list
        )
    )

    latent_change = (
        np.concatenate(
            latent_change_list
        )
    )

    feature_change = (
        np.concatenate(
            feature_change_list
        )
    )

    corr_unc_latent = (
        safe_spearman(
            transition_uncertainty,
            latent_change
        )
    )

    corr_unc_feature = (
        safe_spearman(
            transition_uncertainty,
            feature_change
        )
    )

    transition_validation_df = (
        pd.DataFrame(
            {
                'uncertainty':
                    transition_uncertainty,

                'latent_transition_magnitude':
                    latent_change,

                'feature_transition_magnitude':
                    feature_change
            }
        )
    )

    transition_validation_df[
        'uncertainty_decile'
    ] = (
        pd.qcut(
            transition_validation_df[
                'uncertainty'
            ],
            q=10,
            labels=False,
            duplicates='drop'
        )
        + 1
    )

    transition_decile = (
        transition_validation_df
        .groupby(
            'uncertainty_decile',
            as_index=False
        )
        .agg(
            Mean_Uncertainty=(
                'uncertainty',
                'mean'
            ),

            Mean_Latent_Change=(
                'latent_transition_magnitude',
                'mean'
            ),

            Mean_Feature_Change=(
                'feature_transition_magnitude',
                'mean'
            ),

            N=(
                'uncertainty',
                'size'
            )
        )
    )

    transition_validation_df.to_csv(
        validation_dir
        / "uncertainty_transition_raw.csv",
        index=False
    )

    transition_decile.to_csv(
        validation_dir
        / "uncertainty_transition_deciles.csv",
        index=False
    )

    print(
        "Spearman uncertainty "
        "vs latent change:",
        corr_unc_latent
    )

    print(
        "Spearman uncertainty "
        "vs feature change:",
        corr_unc_feature
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        transition_decile[
            'Mean_Uncertainty'
        ],
        transition_decile[
            'Mean_Latent_Change'
        ],
        marker='o'
    )

    ax.set_xlabel(
        'Mean SDE Diffusion Uncertainty'
    )

    ax.set_ylabel(
        'Mean Latent Transition Magnitude'
    )

    ax.set_title(
        'Uncertainty vs Latent Transition Magnitude'
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        validation_dir
        / "uncertainty_vs_latent_transition.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # ============================================================
    # 2. Uncertainty vs OOD distance
    # ============================================================

    print(
        "\n[2] "
        "Diffusion uncertainty "
        "vs latent OOD distance"
    )

    max_train_samples = 50000

    if (
        len(
            df_rl_train
        )
        >
        max_train_samples
    ):
        df_train_ood = (
            df_rl_train
            .sample(
                n=
                    max_train_samples,
                random_state=42
            )
            .copy()
        )
    else:
        df_train_ood = (
            df_rl_train.copy()
        )

    (
        _,
        z_train_np,
        _,
        _
    ) = (
        validation_encode_dataframe(
            df_train_ood
        )
    )

    (
        _,
        z_test_np,
        _,
        g_test_scaled
    ) = (
        validation_encode_dataframe(
            df_rl_test
        )
    )

    latent_mean = (
        z_train_np.mean(
            axis=0
        )
    )

    covariance = np.cov(
        z_train_np,
        rowvar=False
    )

    covariance = (
        covariance
        + np.eye(
            covariance.shape[0]
        )
        * 1e-4
    )

    inv_covariance = (
        np.linalg.pinv(
            covariance
        )
    )

    centered = (
        z_test_np
        - latent_mean
    )

    mahalanobis_sq = (
        np.einsum(
            'bi,ij,bj->b',
            centered,
            inv_covariance,
            centered
        )
    )

    mahalanobis_distance = (
        np.sqrt(
            np.maximum(
                mahalanobis_sq,
                0
            )
        )
    )

    ood_corr = (
        safe_spearman(
            g_test_scaled,
            mahalanobis_distance
        )
    )

    ood_df = pd.DataFrame(
        {
            'uncertainty':
                g_test_scaled,

            'mahalanobis_distance':
                mahalanobis_distance
        }
    )

    ood_df[
        'uncertainty_decile'
    ] = (
        pd.qcut(
            ood_df[
                'uncertainty'
            ],
            q=10,
            labels=False,
            duplicates='drop'
        )
        + 1
    )

    ood_decile = (
        ood_df
        .groupby(
            'uncertainty_decile',
            as_index=False
        )
        .agg(
            Mean_Uncertainty=(
                'uncertainty',
                'mean'
            ),

            Mean_OOD_Distance=(
                'mahalanobis_distance',
                'mean'
            ),

            Median_OOD_Distance=(
                'mahalanobis_distance',
                'median'
            )
        )
    )

    ood_df.to_csv(
        validation_dir
        / "uncertainty_ood_raw.csv",
        index=False
    )

    ood_decile.to_csv(
        validation_dir
        / "uncertainty_ood_deciles.csv",
        index=False
    )

    print(
        "Spearman uncertainty "
        "vs Mahalanobis OOD:",
        ood_corr
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        ood_decile[
            'Mean_Uncertainty'
        ],
        ood_decile[
            'Mean_OOD_Distance'
        ],
        marker='o'
    )

    ax.set_xlabel(
        'Mean SDE Diffusion Uncertainty'
    )

    ax.set_ylabel(
        'Mean Latent Mahalanobis Distance'
    )

    ax.set_title(
        'Uncertainty vs Latent OOD Distance'
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        validation_dir
        / "uncertainty_vs_ood.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # ============================================================
    # 3. Synthetic perturbation test
    # ============================================================

    print(
        "\n[3] "
        "Synthetic perturbation test"
    )

    rng_validation = (
        np.random.default_rng(
            42
        )
    )

    perturbation_sample_size = min(
        10000,
        len(
            df_rl_test
        )
    )

    perturbation_indices = (
        rng_validation.choice(
            len(
                df_rl_test
            ),
            size=
                perturbation_sample_size,
            replace=False
        )
    )

    df_perturb = (
        df_rl_test
        .iloc[
            perturbation_indices
        ]
        .copy()
    )

    X_base = (
        hmm_module
        .scaler
        .transform(
            df_perturb[
                features_col
            ]
            .values
        )
    )

    noise_levels = [
        0.0,
        0.05,
        0.10,
        0.20,
        0.50,
        1.00
    ]

    perturbation_rows = []

    for noise_level in (
        noise_levels
    ):

        rng_noise = (
            np.random.default_rng(
                42
            )
        )

        X_noise = (
            X_base
            +
            rng_noise.normal(
                loc=0.0,
                scale=
                    noise_level,
                size=
                    X_base.shape
            )
        )

        uncertainty_noise_list = []

        with torch.no_grad():

            for start in range(
                0,
                len(
                    X_noise
                ),
                4096
            ):

                end = min(
                    start + 4096,
                    len(
                        X_noise
                    )
                )

                x = (
                    torch
                    .FloatTensor(
                        X_noise[
                            start:end
                        ]
                    )
                    .to(
                        device
                    )
                )

                z = (
                    validation_get_latent(
                        x
                    )
                )

                (
                    _,
                    g_scaled
                ) = (
                    validation_get_uncertainty(
                        z
                    )
                )

                uncertainty_noise_list.append(
                    g_scaled
                    .cpu()
                    .numpy()
                )

        uncertainty_noise = (
            np.concatenate(
                uncertainty_noise_list
            )
        )

        perturbation_rows.append(
            {
                'Noise_Level':
                    noise_level,

                'Mean_Uncertainty':
                    uncertainty_noise.mean(),

                'Median_Uncertainty':
                    np.median(
                        uncertainty_noise
                    ),

                'Std_Uncertainty':
                    uncertainty_noise.std()
            }
        )

    perturbation_df = (
        pd.DataFrame(
            perturbation_rows
        )
    )

    perturbation_df.to_csv(
        validation_dir
        / "noise_perturbation_uncertainty.csv",
        index=False
    )

    print(
        perturbation_df
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        perturbation_df[
            'Noise_Level'
        ],
        perturbation_df[
            'Mean_Uncertainty'
        ],
        marker='o'
    )

    ax.set_xlabel(
        'Gaussian Noise Level '
        '(Scaled Feature Space)'
    )

    ax.set_ylabel(
        'Mean SDE Diffusion Uncertainty'
    )

    ax.set_title(
        'Perturbation Sensitivity of SDE Uncertainty'
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        validation_dir
        / "noise_vs_uncertainty.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # ============================================================
    # 4. Severity confounding check
    # ============================================================

    print(
        "\n[4] "
        "Severity confounding check"
    )

    severity_rows = []

    severity_variables = [
        'sofa_score',
        'lactate',
        'creatinine',
        'bilirubin',
        'gcs'
    ]

    for column in (
        severity_variables
    ):

        if (
            column
            in policy_df.columns
        ):

            corr_value = (
                safe_spearman(
                    policy_df[
                        'uncertainty'
                    ],
                    policy_df[
                        column
                    ]
                )
            )

            severity_rows.append(
                {
                    'Variable':
                        column,

                    'Spearman_with_Uncertainty':
                        corr_value
                }
            )

    severity_df = pd.DataFrame(
        severity_rows
    )

    severity_df.to_csv(
        validation_dir
        / "uncertainty_severity_correlations.csv",
        index=False
    )

    print(
        severity_df
    )

    mortality_auc = np.nan

    try:

        mortality_auc = (
            roc_auc_score(
                policy_df[
                    'mortality'
                ],
                policy_df[
                    'uncertainty'
                ]
            )
        )

    except Exception:
        pass

    print(
        "Uncertainty -> Mortality AUC:",
        mortality_auc
    )

    mortality_uncertainty = (
        policy_df
        .groupby(
            'mortality'
        )[
            'uncertainty'
        ]
        .agg(
            [
                'count',
                'mean',
                'std',
                'median'
            ]
        )
        .reset_index()
    )

    mortality_uncertainty.to_csv(
        validation_dir
        / "uncertainty_by_mortality.csv",
        index=False
    )

    # ============================================================
    # 5. Counterfactual uncertainty sweep
    #
    # 같은 latent z를 고정하고 uncertainty만 0~1로 변화.
    # High-level이 진짜 uncertainty를 사용하는지 확인.
    # ============================================================

    print(
        "\n[5] "
        "Counterfactual uncertainty sweep"
    )

    cf_sample_size = min(
        20000,
        len(
            df_rl_test
        )
    )

    rng_cf = np.random.default_rng(
        42
    )

    cf_indices = (
        rng_cf.choice(
            len(
                df_rl_test
            ),
            size=
                cf_sample_size,
            replace=False
        )
    )

    df_cf = (
        df_rl_test
        .iloc[
            cf_indices
        ]
        .copy()
    )

    X_cf = (
        hmm_module
        .scaler
        .transform(
            df_cf[
                features_col
            ]
            .values
        )
    )

    z_cf_list = []

    with torch.no_grad():

        for start in range(
            0,
            len(
                X_cf
            ),
            4096
        ):

            end = min(
                start + 4096,
                len(
                    X_cf
                )
            )

            x = (
                torch
                .FloatTensor(
                    X_cf[
                        start:end
                    ]
                )
                .to(
                    device
                )
            )

            z_cf_list.append(
                validation_get_latent(
                    x
                )
                .cpu()
            )

    z_cf = torch.cat(
        z_cf_list,
        dim=0
    ).to(
        device
    )

    uncertainty_grid = (
        np.linspace(
            0.0,
            1.0,
            21
        )
    )

    counterfactual_rows = []

    with torch.no_grad():

        for u in (
            uncertainty_grid
        ):

            u_tensor = (
                torch.full(
                    (
                        z_cf.shape[0],
                    ),
                    float(
                        u
                    ),
                    device=device
                )
            )

            high_values = (
                high_net(
                    z_cf,
                    u_tensor
                )
            )

            high_softmax = (
                torch.softmax(
                    high_values,
                    dim=1
                )
            )

            prospect_preference = (
                high_softmax[
                    :,
                    1
                ]
            )

            prospect_choice = (
                high_values
                .argmax(
                    dim=1
                )
            )

            counterfactual_rows.append(
                {
                    'Forced_Uncertainty':
                        u,

                    'Mean_Prospect_Softmax_Preference':
                        (
                            prospect_preference
                            .mean()
                            .item()
                        ),

                    'Prospect_Argmax_Ratio':
                        (
                            (
                                prospect_choice
                                == 1
                            )
                            .float()
                            .mean()
                            .item()
                        ),

                    'Mean_Q_Option_Value':
                        (
                            high_values[
                                :,
                                0
                            ]
                            .mean()
                            .item()
                        ),

                    'Mean_Prospect_Option_Value':
                        (
                            high_values[
                                :,
                                1
                            ]
                            .mean()
                            .item()
                        )
                }
            )

    counterfactual_df = pd.DataFrame(
        counterfactual_rows
    )

    counterfactual_df.to_csv(
        validation_dir
        / "counterfactual_uncertainty_sweep.csv",
        index=False
    )

    print(
        counterfactual_df
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        counterfactual_df[
            'Forced_Uncertainty'
        ],
        counterfactual_df[
            'Prospect_Argmax_Ratio'
        ],
        marker='o'
    )

    ax.set_xlabel(
        'Forced SDE Uncertainty'
    )

    ax.set_ylabel(
        'Prospect Selection Ratio'
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        'Counterfactual Effect of Uncertainty '
        'on High-Level Selection'
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        validation_dir
        / "counterfactual_uncertainty_selection.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(
        fig
    )

    # ============================================================
    # 6. Shuffle / Fixed uncertainty ablation
    # ============================================================

    print(
        "\n[6] "
        "Uncertainty ablation"
    )

    policy_eval_df = (
        df_rl_test
        .sort_values(
            [
                'stay_id',
                'charttime'
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    X_policy = (
        hmm_module
        .scaler
        .transform(
            policy_eval_df[
                features_col
            ]
            .values
        )
    )

    original_option_list = []
    fixed_option_list = []
    shuffled_option_list = []

    original_action_list = []
    fixed_action_list = []
    shuffled_action_list = []

    true_action = (
        policy_eval_df[
            'action_to_next'
        ]
        .astype(int)
        .to_numpy()
    )

    rng_shuffle = (
        np.random.default_rng(
            42
        )
    )

    with torch.no_grad():

        for start in range(
            0,
            len(
                X_policy
            ),
            4096
        ):

            end = min(
                start + 4096,
                len(
                    X_policy
                )
            )

            x = (
                torch
                .FloatTensor(
                    X_policy[
                        start:end
                    ]
                )
                .to(
                    device
                )
            )

            z = (
                validation_get_latent(
                    x
                )
            )

            (
                _,
                g_original
            ) = (
                validation_get_uncertainty(
                    z
                )
            )

            q_action = (
                q_net(
                    z
                )
                .argmax(
                    dim=1
                )
            )

            p_action = (
                p_net(
                    z
                )
                .argmax(
                    dim=1
                )
            )

            option_original = (
                high_net(
                    z,
                    g_original
                )
                .argmax(
                    dim=1
                )
            )

            g_fixed = (
                torch.full_like(
                    g_original,
                    0.5
                )
            )

            option_fixed = (
                high_net(
                    z,
                    g_fixed
                )
                .argmax(
                    dim=1
                )
            )

            shuffled_indices = (
                rng_shuffle
                .permutation(
                    len(
                        g_original
                    )
                )
            )

            g_shuffled = (
                g_original[
                    torch.LongTensor(
                        shuffled_indices
                    )
                    .to(
                        device
                    )
                ]
            )

            option_shuffled = (
                high_net(
                    z,
                    g_shuffled
                )
                .argmax(
                    dim=1
                )
            )

            action_original = (
                torch.where(
                    option_original
                    == 0,
                    q_action,
                    p_action
                )
            )

            action_fixed = (
                torch.where(
                    option_fixed
                    == 0,
                    q_action,
                    p_action
                )
            )

            action_shuffled = (
                torch.where(
                    option_shuffled
                    == 0,
                    q_action,
                    p_action
                )
            )

            original_option_list.append(
                option_original
                .cpu()
                .numpy()
            )

            fixed_option_list.append(
                option_fixed
                .cpu()
                .numpy()
            )

            shuffled_option_list.append(
                option_shuffled
                .cpu()
                .numpy()
            )

            original_action_list.append(
                action_original
                .cpu()
                .numpy()
            )

            fixed_action_list.append(
                action_fixed
                .cpu()
                .numpy()
            )

            shuffled_action_list.append(
                action_shuffled
                .cpu()
                .numpy()
            )

    original_option = (
        np.concatenate(
            original_option_list
        )
    )

    fixed_option = (
        np.concatenate(
            fixed_option_list
        )
    )

    shuffled_option = (
        np.concatenate(
            shuffled_option_list
        )
    )

    original_action = (
        np.concatenate(
            original_action_list
        )
    )

    fixed_action = (
        np.concatenate(
            fixed_action_list
        )
    )

    shuffled_action = (
        np.concatenate(
            shuffled_action_list
        )
    )

    uncertainty_ablation = (
        pd.DataFrame(
            [
                {
                    'Setting':
                        'Original',

                    'Prospect_Ratio':
                        (
                            original_option
                            == 1
                        ).mean(),

                    'Clinician_Agreement':
                        (
                            original_action
                            == true_action
                        ).mean(),

                    'Option_Change_vs_Original':
                        0.0,

                    'Action_Change_vs_Original':
                        0.0
                },

                {
                    'Setting':
                        'Fixed_0.5',

                    'Prospect_Ratio':
                        (
                            fixed_option
                            == 1
                        ).mean(),

                    'Clinician_Agreement':
                        (
                            fixed_action
                            == true_action
                        ).mean(),

                    'Option_Change_vs_Original':
                        (
                            fixed_option
                            != original_option
                        ).mean(),

                    'Action_Change_vs_Original':
                        (
                            fixed_action
                            != original_action
                        ).mean()
                },

                {
                    'Setting':
                        'Shuffled',

                    'Prospect_Ratio':
                        (
                            shuffled_option
                            == 1
                        ).mean(),

                    'Clinician_Agreement':
                        (
                            shuffled_action
                            == true_action
                        ).mean(),

                    'Option_Change_vs_Original':
                        (
                            shuffled_option
                            != original_option
                        ).mean(),

                    'Action_Change_vs_Original':
                        (
                            shuffled_action
                            != original_action
                        ).mean()
                }
            ]
        )
    )

    uncertainty_ablation.to_csv(
        validation_dir
        / "uncertainty_ablation.csv",
        index=False
    )

    print(
        uncertainty_ablation
    )

    # ============================================================
    # 7. Independent uncertainty-option alignment metrics
    # ============================================================

    print(
        "\n[7] "
        "Independent uncertainty-option alignment"
    )

    independent_spearman = (
        safe_spearman(
            policy_df[
                'uncertainty'
            ],
            policy_df[
                'high_option'
            ]
        )
    )

    try:

        uncertainty_option_auc = (
            roc_auc_score(
                policy_df[
                    'high_option'
                ],
                policy_df[
                    'uncertainty'
                ]
            )
        )

    except Exception:

        uncertainty_option_auc = (
            np.nan
        )

    mean_q_uncertainty = (
        policy_df.loc[
            policy_df[
                'high_option'
            ]
            == 0,
            'uncertainty'
        ]
        .mean()
    )

    mean_p_uncertainty = (
        policy_df.loc[
            policy_df[
                'high_option'
            ]
            == 1,
            'uncertainty'
        ]
        .mean()
    )

    independent_gap = (
        mean_p_uncertainty
        - mean_q_uncertainty
    )

    alignment_metrics = (
        pd.DataFrame(
            [
                {
                    'Metric':
                        'Spearman(Uncertainty, Option)',

                    'Value':
                        independent_spearman
                },

                {
                    'Metric':
                        'AUC(Uncertainty -> Prospect)',

                    'Value':
                        uncertainty_option_auc
                },

                {
                    'Metric':
                        'Mean Uncertainty Q',

                    'Value':
                        mean_q_uncertainty
                },

                {
                    'Metric':
                        'Mean Uncertainty Prospect',

                    'Value':
                        mean_p_uncertainty
                },

                {
                    'Metric':
                        'Uncertainty Gap',

                    'Value':
                        independent_gap
                }
            ]
        )
    )

    alignment_metrics.to_csv(
        validation_dir
        / "independent_alignment_metrics.csv",
        index=False
    )

    print(
        alignment_metrics
    )

    # ============================================================
    # 8. Patient-level bootstrap confidence intervals
    # ============================================================

    print(
        "\n[8] "
        "Patient-level bootstrap confidence intervals"
    )

    bootstrap_rng = (
        np.random.default_rng(
            42
        )
    )

    unique_stays = (
        policy_df[
            'stay_id'
        ]
        .unique()
    )

    n_bootstrap = 1000

    bootstrap_rows = []

    grouped_indices = {
        stay_id:
            np.where(
                policy_df[
                    'stay_id'
                ]
                .to_numpy()
                == stay_id
            )[0]
        for stay_id
        in unique_stays
    }

    clinician_array = (
        policy_df[
            'clinician_action'
        ]
        .to_numpy()
    )

    hrl_array = (
        policy_df[
            'hrl_action'
        ]
        .to_numpy()
    )

    q_array = (
        policy_df[
            'q_action'
        ]
        .to_numpy()
    )

    p_array = (
        policy_df[
            'prospect_action'
        ]
        .to_numpy()
    )

    unc_array = (
        policy_df[
            'uncertainty'
        ]
        .to_numpy()
    )

    option_array = (
        policy_df[
            'high_option'
        ]
        .to_numpy()
    )

    for b in range(
        n_bootstrap
    ):

        sampled_stays = (
            bootstrap_rng.choice(
                unique_stays,
                size=
                    len(
                        unique_stays
                    ),
                replace=True
            )
        )

        sampled_indices = (
            np.concatenate(
                [
                    grouped_indices[
                        stay_id
                    ]
                    for stay_id
                    in sampled_stays
                ]
            )
        )

        clinician_b = (
            clinician_array[
                sampled_indices
            ]
        )

        hrl_b = (
            hrl_array[
                sampled_indices
            ]
        )

        q_b = (
            q_array[
                sampled_indices
            ]
        )

        p_b = (
            p_array[
                sampled_indices
            ]
        )

        unc_b = (
            unc_array[
                sampled_indices
            ]
        )

        option_b = (
            option_array[
                sampled_indices
            ]
        )

        q_mask = (
            option_b == 0
        )

        p_mask = (
            option_b == 1
        )

        if (
            q_mask.sum() > 0
            and
            p_mask.sum() > 0
        ):

            unc_gap_b = (
                unc_b[
                    p_mask
                ].mean()
                -
                unc_b[
                    q_mask
                ].mean()
            )

        else:

            unc_gap_b = np.nan

        bootstrap_rows.append(
            {
                'HRL_minus_Q_Agreement':
                    (
                        (
                            hrl_b
                            == clinician_b
                        ).mean()
                        -
                        (
                            q_b
                            == clinician_b
                        ).mean()
                    ),

                'HRL_minus_Prospect_Agreement':
                    (
                        (
                            hrl_b
                            == clinician_b
                        ).mean()
                        -
                        (
                            p_b
                            == clinician_b
                        ).mean()
                    ),

                'Uncertainty_Gap':
                    unc_gap_b
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap_df.to_csv(
        validation_dir
        / "bootstrap_raw.csv",
        index=False
    )

    bootstrap_summary_rows = []

    for column in [
        'HRL_minus_Q_Agreement',
        'HRL_minus_Prospect_Agreement',
        'Uncertainty_Gap'
    ]:

        values = (
            bootstrap_df[
                column
            ]
            .dropna()
            .to_numpy()
        )

        bootstrap_summary_rows.append(
            {
                'Metric':
                    column,

                'Mean':
                    np.mean(
                        values
                    ),

                'CI_2.5':
                    np.percentile(
                        values,
                        2.5
                    ),

                'CI_97.5':
                    np.percentile(
                        values,
                        97.5
                    )
            }
        )

    bootstrap_summary = pd.DataFrame(
        bootstrap_summary_rows
    )

    bootstrap_summary.to_csv(
        validation_dir
        / "bootstrap_summary.csv",
        index=False
    )

    print(
        bootstrap_summary
    )

    # ============================================================
    # 9. PD-WIS
    # ============================================================

    print(
        "\n[9] "
        "Per-Decision Weighted Importance Sampling"
    )

    try:

        behavior_model = (
            result_visualizer
            .train_behavior_policy(
                df_train=
                    df_rl_train,
                scaler=
                    hmm_module.scaler,
                features_col=
                    features_col,
                encoder_module=
                    encoder_module,
                action_dim=25,
                epochs=5
            )
        )

        (
            pdwis_df,
            behavior_probs,
            q_probs,
            p_probs,
            hrl_probs
        ) = (
            result_visualizer
            ._policy_probabilities(
                df=
                    df_rl_test,
                scaler=
                    hmm_module.scaler,
                features_col=
                    features_col,
                encoder_module=
                    encoder_module,
                q_net=
                    q_net,
                p_net=
                    p_net,
                high_net=
                    high_net,
                behavior_model=
                    behavior_model,
                g_mean=
                    g_mean,
                g_std=
                    g_std,
                temperature=1.0
            )
        )

        pdwis_actions = (
            pdwis_df[
                'action_to_next'
            ]
            .astype(int)
            .to_numpy()
        )

        pdwis_row_idx = (
            np.arange(
                len(
                    pdwis_df
                )
            )
        )

        behavior_action_prob = (
            behavior_probs[
                pdwis_row_idx,
                pdwis_actions
            ]
        )

        behavior_action_prob = (
            np.clip(
                behavior_action_prob,
                1e-6,
                1.0
            )
        )

        target_policy_probs = {
            'HRL':
                hrl_probs[
                    pdwis_row_idx,
                    pdwis_actions
                ],

            'Only Q':
                q_probs[
                    pdwis_row_idx,
                    pdwis_actions
                ],

            'Only Prospect':
                p_probs[
                    pdwis_row_idx,
                    pdwis_actions
                ]
        }

        rewards_pdwis = (
            result_visualizer
            ._make_reward(
                pdwis_df
            )
        )

        pdwis_base = (
            pdwis_df[
                [
                    'stay_id',
                    'charttime'
                ]
            ]
            .copy()
        )

        pdwis_base[
            'reward'
        ] = rewards_pdwis

        pdwis_base[
            'behavior_prob'
        ] = (
            behavior_action_prob
        )

        pdwis_summary_rows = []

        gamma_pdwis = gamma

        for (
            policy_name,
            target_prob
        ) in (
            target_policy_probs
            .items()
        ):

            temp = (
                pdwis_base
                .copy()
            )

            temp[
                'target_prob'
            ] = (
                np.clip(
                    target_prob,
                    1e-6,
                    1.0
                )
            )

            temp = (
                temp
                .sort_values(
                    [
                        'stay_id',
                        'charttime'
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            trajectory_data = []

            max_t = 0

            for stay_id, group in (
                temp.groupby(
                    'stay_id',
                    sort=False
                )
            ):

                reward_seq = (
                    group[
                        'reward'
                    ]
                    .to_numpy()
                )

                target_seq = (
                    group[
                        'target_prob'
                    ]
                    .to_numpy()
                )

                behavior_seq = (
                    group[
                        'behavior_prob'
                    ]
                    .to_numpy()
                )

                log_ratio_seq = (
                    np.log(
                        target_seq
                    )
                    -
                    np.log(
                        behavior_seq
                    )
                )

                cumulative_log_ratio = (
                    np.cumsum(
                        log_ratio_seq
                    )
                )

                trajectory_data.append(
                    (
                        reward_seq,
                        cumulative_log_ratio
                    )
                )

                max_t = max(
                    max_t,
                    len(
                        reward_seq
                    )
                )

            pdwis_value = 0.0

            ess_by_t = []

            for t in range(
                max_t
            ):

                rewards_t = []
                log_weights_t = []

                for (
                    reward_seq,
                    log_weight_seq
                ) in trajectory_data:

                    if (
                        t
                        <
                        len(
                            reward_seq
                        )
                    ):

                        rewards_t.append(
                            reward_seq[
                                t
                            ]
                        )

                        log_weights_t.append(
                            log_weight_seq[
                                t
                            ]
                        )

                if (
                    len(
                        rewards_t
                    )
                    == 0
                ):
                    continue

                rewards_t = np.asarray(
                    rewards_t
                )

                log_weights_t = (
                    np.asarray(
                        log_weights_t
                    )
                )

                max_log = (
                    np.max(
                        log_weights_t
                    )
                )

                weights_t = (
                    np.exp(
                        log_weights_t
                        - max_log
                    )
                )

                weights_sum = (
                    weights_t.sum()
                )

                if (
                    weights_sum
                    <= 0
                ):
                    continue

                normalized_weights = (
                    weights_t
                    / weights_sum
                )

                weighted_reward = (
                    np.sum(
                        normalized_weights
                        * rewards_t
                    )
                )

                pdwis_value += (
                    (
                        gamma_pdwis
                        ** t
                    )
                    * weighted_reward
                )

                ess_t = (
                    1.0
                    /
                    np.sum(
                        normalized_weights
                        ** 2
                    )
                )

                ess_by_t.append(
                    ess_t
                )

            pdwis_summary_rows.append(
                {
                    'Policy':
                        policy_name,

                    'PDWIS':
                        pdwis_value,

                    'Min_ESS':
                        (
                            np.min(
                                ess_by_t
                            )
                            if len(
                                ess_by_t
                            ) > 0
                            else np.nan
                        ),

                    'Median_ESS':
                        (
                            np.median(
                                ess_by_t
                            )
                            if len(
                                ess_by_t
                            ) > 0
                            else np.nan
                        ),

                    'Mean_ESS':
                        (
                            np.mean(
                                ess_by_t
                            )
                            if len(
                                ess_by_t
                            ) > 0
                            else np.nan
                        ),

                    'Max_Horizon':
                        max_t
                }
            )

        pdwis_summary = (
            pd.DataFrame(
                pdwis_summary_rows
            )
        )

        pdwis_summary.to_csv(
            validation_dir
            / "pdwis_summary.csv",
            index=False
        )

        print(
            pdwis_summary
        )

    except Exception as e:

        print(
            "PD-WIS calculation failed:"
        )

        print(
            e
        )

    # ============================================================
    # 10. HMM state vs uncertainty
    # ============================================================

    print(
        "\n[10] "
        "HMM state vs uncertainty"
    )

    try:

        hmm_validation_df = (
            policy_df.copy()
        )

        hmm_validation_df[
            'hmm_state'
        ] = (
            hmm_module.predict(
                hmm_validation_df,
                features_col
            )
        )

        hmm_uncertainty = (
            hmm_validation_df
            .groupby(
                'hmm_state',
                as_index=False
            )
            .agg(
                Mean_Uncertainty=(
                    'uncertainty',
                    'mean'
                ),

                Median_Uncertainty=(
                    'uncertainty',
                    'median'
                ),

                Prospect_Ratio=(
                    'high_option',
                    lambda x:
                        (
                            x == 1
                        ).mean()
                ),

                Mean_SOFA=(
                    'sofa_score',
                    'mean'
                ),

                N=(
                    'stay_id',
                    'size'
                )
            )
        )

        hmm_uncertainty.to_csv(
            validation_dir
            / "hmm_state_uncertainty.csv",
            index=False
        )

        print(
            hmm_uncertainty
        )

    except Exception as e:

        print(
            "HMM uncertainty validation failed:"
        )

        print(
            e
        )

    # ============================================================
    # 11. Validation summary
    # ============================================================

    validation_summary = (
        pd.DataFrame(
            [
                {
                    'Uncertainty_vs_LatentChange_Spearman':
                        corr_unc_latent,

                    'Uncertainty_vs_FeatureChange_Spearman':
                        corr_unc_feature,

                    'Uncertainty_vs_OOD_Spearman':
                        ood_corr,

                    'Uncertainty_Option_Spearman':
                        independent_spearman,

                    'Uncertainty_Option_AUC':
                        uncertainty_option_auc,

                    'Mean_Q_Uncertainty':
                        mean_q_uncertainty,

                    'Mean_Prospect_Uncertainty':
                        mean_p_uncertainty,

                    'Uncertainty_Gap':
                        independent_gap,

                    'Uncertainty_Mortality_AUC':
                        mortality_auc,

                    'Fixed_Uncertainty_Option_Change':
                        (
                            fixed_option
                            != original_option
                        ).mean(),

                    'Shuffled_Uncertainty_Option_Change':
                        (
                            shuffled_option
                            != original_option
                        ).mean(),

                    'Fixed_Uncertainty_Action_Change':
                        (
                            fixed_action
                            != original_action
                        ).mean(),

                    'Shuffled_Uncertainty_Action_Change':
                        (
                            shuffled_action
                            != original_action
                        ).mean()
                }
            ]
        )
    )

    validation_summary.to_csv(
        validation_dir
        / "validation_summary.csv",
        index=False
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        validation_summary.T
    )

    print(
        "\nValidation results saved to:"
    )

    print(
        validation_dir
    )
    # ============================================================
    # STAGE 1 VALIDATION
    # H1: Severity -> Diffusion
    # H2: Severity -> 4h next-state prediction error
    # H3: Diffusion -> prediction error | Severity + OOD
    # Prediction error is measured in the SAME scaled feature space
    # used by the encoder training target: Decoder(SDE(z_t)) vs X_{t+1}.
    # ============================================================
    import statsmodels.api as sm
    from scipy.stats import spearmanr, pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("STAGE 1 : SDE UNCERTAINTY VALIDATION (CORRECTED)")
    print("=" * 60)

    MC_SAMPLES = 30
    MAX_VALIDATION_TRANSITIONS = 15000
    MAX_OOD_TRAIN_SAMPLES = 50000
    RANDOM_SEED = 42
    stage1_dir = PROJECT_ROOT / "results" / "stage1_sde_validation_corrected"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    encoder_module.encoder.eval(); encoder_module.sde.eval(); encoder_module.decoder.eval()

    # RL data is already restricted to onset 0~analysis_window_hours and exact 4h transitions.
    df_transition = df_rl_test.copy().sort_values(['stay_id', 'charttime']).reset_index(drop=True)
    df_transition = df_transition[df_transition['hours_from_onset'].between(0, analysis_window_hours, inclusive='both')].copy()
    df_transition = df_transition[np.isclose(df_transition['transition_hours'], interval)].copy()
    df_transition = df_transition.dropna(subset=features_col + next_features_cols + ['sofa_score', 'stay_id']).reset_index(drop=True)
    if len(df_transition) > MAX_VALIDATION_TRANSITIONS:
        df_transition = df_transition.sample(MAX_VALIDATION_TRANSITIONS, random_state=RANDOM_SEED).sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    print(f"Window: onset 0-{analysis_window_hours}h")
    print(f"Valid test transitions used: {len(df_transition)}")
    print(f"Unique test stays: {df_transition['stay_id'].nunique()}")
    print(f"Observed max transition horizon per stay: {df_transition.groupby('stay_id').size().max()} steps")

    def encode_mu(x_np, batch_size=4096):
        out = []
        with torch.no_grad():
            for start_idx in range(0, len(x_np), batch_size):
                x = torch.as_tensor(x_np[start_idx:start_idx + batch_size], dtype=torch.float32, device=device)
                mu, _ = encoder_module.encoder(x)
                out.append(mu.cpu().numpy())
        return np.concatenate(out, axis=0)

    def diffusion_from_z(z_np, batch_size=4096):
        raw, scaled = [], []
        with torch.no_grad():
            for start_idx in range(0, len(z_np), batch_size):
                z = torch.as_tensor(z_np[start_idx:start_idx + batch_size], dtype=torch.float32, device=device)
                t0 = torch.zeros_like(z[:, :1])
                g_val = encoder_module.sde.g_net(torch.cat([t0, z], dim=1)) + 1e-3
                g_raw = g_val.max(dim=1).values
                g_scaled = torch.sigmoid((g_raw - g_mean) / (g_std + 1e-6))
                raw.append(g_raw.cpu().numpy()); scaled.append(g_scaled.cpu().numpy())
        return np.concatenate(raw), np.concatenate(scaled)

    X_current = hmm_module.scaler.transform(df_transition[features_col].values)
    X_next = hmm_module.scaler.transform(df_transition[next_features_cols].values)
    z_current = encode_mu(X_current)
    g_raw, g_scaled = diffusion_from_z(z_current)

    # ------------------------------------------------------------
    # Monte-Carlo one-step SDE prediction.
    # z_t = encoder mean is held fixed so dispersion comes from SDE noise itself.
    # Every SDE sample is decoded back to the scaled clinical feature space.
    # ------------------------------------------------------------
    print(f"Running {MC_SAMPLES}x SDE Monte-Carlo one-step prediction...")
    pred_mean_list, pred_error_list, pred_dispersion_list = [], [], []
    SDE_BATCH_SIZE = 256

    with torch.no_grad():
        for start_idx in range(0, len(z_current), SDE_BATCH_SIZE):
            end_idx = min(start_idx + SDE_BATCH_SIZE, len(z_current))
            z = torch.as_tensor(z_current[start_idx:end_idx], dtype=torch.float32, device=device)
            x_next = torch.as_tensor(X_next[start_idx:end_idx], dtype=torch.float32, device=device)
            b = z.shape[0]
            z_repeat = z[:, None, :].expand(b, MC_SAMPLES, z.shape[1]).reshape(b * MC_SAMPLES, z.shape[1]).contiguous()
            z_traj, _ = encoder_module.sde(z_repeat, encoder_module.ts)
            z_pred = z_traj[-1]
            x_pred = encoder_module.decoder(z_pred).reshape(b, MC_SAMPLES, len(features_col))
            x_pred_mean = x_pred.mean(dim=1)
            pred_error = torch.sqrt(torch.mean((x_next - x_pred_mean) ** 2, dim=1))
            pred_std = x_pred.std(dim=1, unbiased=False)
            pred_dispersion = torch.sqrt(torch.mean(pred_std ** 2, dim=1))
            pred_mean_list.append(x_pred_mean.cpu().numpy())
            pred_error_list.append(pred_error.cpu().numpy())
            pred_dispersion_list.append(pred_dispersion.cpu().numpy())

    X_pred_mean = np.concatenate(pred_mean_list, axis=0)
    prediction_error = np.concatenate(pred_error_list)
    predictive_dispersion = np.concatenate(pred_dispersion_list)

    # Per-feature standardized RMSE, useful for checking whether one feature dominates.
    per_feature_rmse = np.sqrt(np.mean((X_next - X_pred_mean) ** 2, axis=0))
    per_feature_df = pd.DataFrame({'Feature': features_col, 'Standardized_RMSE': per_feature_rmse})
    per_feature_df.to_csv(stage1_dir / 'per_feature_prediction_rmse.csv', index=False)

    # ------------------------------------------------------------
    # Latent OOD: Mahalanobis distance from RL-train latent distribution.
    # Train/test are now on the same 0~96h horizon.
    # ------------------------------------------------------------
    df_ood_train = df_rl_train.copy()
    if len(df_ood_train) > MAX_OOD_TRAIN_SAMPLES:
        df_ood_train = df_ood_train.sample(MAX_OOD_TRAIN_SAMPLES, random_state=RANDOM_SEED)
    X_train_ood = hmm_module.scaler.transform(df_ood_train[features_col].values)
    z_train_ood = encode_mu(X_train_ood)
    latent_mean = z_train_ood.mean(axis=0)
    latent_cov = np.cov(z_train_ood, rowvar=False) + np.eye(z_train_ood.shape[1]) * 1e-4
    inv_latent_cov = np.linalg.pinv(latent_cov)
    centered = z_current - latent_mean
    ood_distance = np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', centered, inv_latent_cov, centered), 0))

    validation_df = df_transition[['stay_id', 'charttime', 'hours_from_onset', 'sofa_score']].copy()
    validation_df['diffusion_raw'] = g_raw
    validation_df['diffusion_scaled'] = g_scaled
    validation_df['prediction_error'] = prediction_error
    validation_df['predictive_dispersion'] = predictive_dispersion
    validation_df['ood_distance'] = ood_distance
    validation_df.to_csv(stage1_dir / 'stage1_transition_validation.csv', index=False)

    # ------------------------------------------------------------
    # H1: Severity -> Diffusion
    # ------------------------------------------------------------
    h1_rho, h1_p = spearmanr(validation_df['sofa_score'], validation_df['diffusion_raw'], nan_policy='omit')
    validation_df['SOFA_Group'] = pd.cut(validation_df['sofa_score'], [-np.inf, 3, 6, 9, 12, np.inf], labels=['0-3', '4-6', '7-9', '10-12', '13+'])
    h1_group = validation_df.groupby('SOFA_Group', observed=True).agg(Mean_SOFA=('sofa_score', 'mean'), Mean_Diffusion=('diffusion_raw', 'mean'), Median_Diffusion=('diffusion_raw', 'median'), Mean_Scaled_Diffusion=('diffusion_scaled', 'mean'), N=('stay_id', 'size')).reset_index()
    h1_group.to_csv(stage1_dir / 'H1_severity_vs_diffusion.csv', index=False)

    # ------------------------------------------------------------
    # H2: Severity -> decoder-space next-state prediction error
    # ------------------------------------------------------------
    h2_rho, h2_p = spearmanr(validation_df['sofa_score'], validation_df['prediction_error'], nan_policy='omit')
    h2_group = validation_df.groupby('SOFA_Group', observed=True).agg(Mean_SOFA=('sofa_score', 'mean'), Mean_Prediction_Error=('prediction_error', 'mean'), Median_Prediction_Error=('prediction_error', 'median'), Mean_Predictive_Dispersion=('predictive_dispersion', 'mean'), N=('stay_id', 'size')).reset_index()
    h2_group.to_csv(stage1_dir / 'H2_severity_vs_prediction_error.csv', index=False)

    # Additional calibration diagnostics.
    diffusion_error_rho, diffusion_error_p = spearmanr(validation_df['diffusion_raw'], validation_df['prediction_error'], nan_policy='omit')
    diffusion_dispersion_rho, diffusion_dispersion_p = spearmanr(validation_df['diffusion_raw'], validation_df['predictive_dispersion'], nan_policy='omit')
    dispersion_error_rho, dispersion_error_p = spearmanr(validation_df['predictive_dispersion'], validation_df['prediction_error'], nan_policy='omit')
    diffusion_ood_rho, diffusion_ood_p = spearmanr(validation_df['diffusion_raw'], validation_df['ood_distance'], nan_policy='omit')

    # ------------------------------------------------------------
    # H3: Error ~ Diffusion + Severity + OOD
    # Cluster-robust SE by stay_id because each stay contributes repeated transitions.
    # ------------------------------------------------------------
    regression_df = validation_df[['stay_id', 'prediction_error', 'diffusion_raw', 'sofa_score', 'ood_distance']].replace([np.inf, -np.inf], np.nan).dropna().copy()
    scaler_reg = StandardScaler()
    zvals = scaler_reg.fit_transform(regression_df[['prediction_error', 'diffusion_raw', 'sofa_score', 'ood_distance']])
    regression_df[['error_z', 'diffusion_z', 'severity_z', 'ood_z']] = zvals

    X_unadjusted = sm.add_constant(regression_df[['diffusion_z']])
    model_unadjusted = sm.OLS(regression_df['error_z'], X_unadjusted).fit(cov_type='cluster', cov_kwds={'groups': regression_df['stay_id']})
    X_adjusted = sm.add_constant(regression_df[['diffusion_z', 'severity_z', 'ood_z']])
    model_adjusted = sm.OLS(regression_df['error_z'], X_adjusted).fit(cov_type='cluster', cov_kwds={'groups': regression_df['stay_id']})

    def model_frame(model, name):
        ci = model.conf_int()
        return pd.DataFrame({'Model': name, 'Variable': model.params.index, 'Beta': model.params.values, 'Std_Error': model.bse.values, 'P_Value': model.pvalues.values, 'CI_2.5': ci[0].values, 'CI_97.5': ci[1].values})

    regression_results = pd.concat([model_frame(model_unadjusted, 'Unadjusted'), model_frame(model_adjusted, 'Adjusted_Severity_OOD')], ignore_index=True)
    regression_results.to_csv(stage1_dir / 'H3_diffusion_prediction_error_regression.csv', index=False)

    # Partial Spearman: rank-transform all variables, regress out severity/OOD, then correlate residuals.
    rank_df = regression_df[['diffusion_raw', 'prediction_error', 'sofa_score', 'ood_distance']].rank(method='average')
    controls = rank_df[['sofa_score', 'ood_distance']].values
    diffusion_resid = rank_df['diffusion_raw'].values - LinearRegression().fit(controls, rank_df['diffusion_raw'].values).predict(controls)
    error_resid = rank_df['prediction_error'].values - LinearRegression().fit(controls, rank_df['prediction_error'].values).predict(controls)
    partial_rho, partial_p = pearsonr(diffusion_resid, error_resid)

    adjusted_beta = model_adjusted.params['diffusion_z']
    adjusted_p = model_adjusted.pvalues['diffusion_z']
    adjusted_ci = model_adjusted.conf_int().loc['diffusion_z']

    summary = pd.DataFrame([{
        'H1_Severity_Diffusion_Spearman': h1_rho, 'H1_P_Value': h1_p,
        'H2_Severity_PredictionError_Spearman': h2_rho, 'H2_P_Value': h2_p,
        'Diffusion_PredictionError_Spearman': diffusion_error_rho, 'Diffusion_PredictionError_P_Value': diffusion_error_p,
        'Diffusion_PredictiveDispersion_Spearman': diffusion_dispersion_rho, 'Diffusion_PredictiveDispersion_P_Value': diffusion_dispersion_p,
        'PredictiveDispersion_Error_Spearman': dispersion_error_rho, 'PredictiveDispersion_Error_P_Value': dispersion_error_p,
        'Diffusion_OOD_Spearman': diffusion_ood_rho, 'Diffusion_OOD_P_Value': diffusion_ood_p,
        'H3_Adjusted_Diffusion_Beta': adjusted_beta, 'H3_Adjusted_Diffusion_P_Value': adjusted_p,
        'H3_Adjusted_Diffusion_CI_Low': adjusted_ci.iloc[0], 'H3_Adjusted_Diffusion_CI_High': adjusted_ci.iloc[1],
        'H3_Partial_Spearman': partial_rho, 'H3_Partial_P_Value': partial_p,
        'Mean_Prediction_Error': validation_df['prediction_error'].mean(), 'Mean_Predictive_Dispersion': validation_df['predictive_dispersion'].mean(),
        'Transitions': len(validation_df), 'Unique_Stays': validation_df['stay_id'].nunique(), 'Window_Hours': analysis_window_hours, 'MC_Samples': MC_SAMPLES
    }])
    summary.to_csv(stage1_dir / 'STAGE1_FINAL_SUMMARY.csv', index=False)

    print("\n" + "=" * 60)
    print("H1 : SEVERITY -> DIFFUSION")
    print("=" * 60)
    print(f"Spearman rho={h1_rho:.6f}, p={h1_p:.3e}")
    print(h1_group.to_string(index=False))

    print("\n" + "=" * 60)
    print("H2 : SEVERITY -> 4H NEXT-STATE PREDICTION ERROR")
    print("=" * 60)
    print(f"Spearman rho={h2_rho:.6f}, p={h2_p:.3e}")
    print(h2_group.to_string(index=False))
    print("\nPer-feature standardized RMSE")
    print(per_feature_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("UNCERTAINTY CALIBRATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Diffusion vs Prediction Error: rho={diffusion_error_rho:.6f}, p={diffusion_error_p:.3e}")
    print(f"Diffusion vs Predictive Dispersion: rho={diffusion_dispersion_rho:.6f}, p={diffusion_dispersion_p:.3e}")
    print(f"Predictive Dispersion vs Prediction Error: rho={dispersion_error_rho:.6f}, p={dispersion_error_p:.3e}")
    print(f"Diffusion vs OOD: rho={diffusion_ood_rho:.6f}, p={diffusion_ood_p:.3e}")

    print("\n" + "=" * 60)
    print("H3 : DIFFUSION -> PREDICTION ERROR | SEVERITY + OOD")
    print("=" * 60)
    print(model_adjusted.summary())
    print(f"Adjusted diffusion beta={adjusted_beta:.6f}, p={adjusted_p:.6f}, 95% CI=[{adjusted_ci.iloc[0]:.6f}, {adjusted_ci.iloc[1]:.6f}]")
    print(f"Partial Spearman={partial_rho:.6f}, p={partial_p:.3e}")

    print("\n" + "=" * 60)
    print("STAGE 1 FINAL SUMMARY")
    print("=" * 60)
    print(summary.T)
    print(f"\nResults saved to: {stage1_dir}")

