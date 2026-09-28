from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Preprocessing.preprocessing import target_cohort, states_preprocessor, action_preprocessor, sofa, survival_labeler
from Pretraining.encoder import SepsisEncoder
from Pretraining.hmm import SepsisHMM
from Pretraining.sampling import Sample
from db_conn import db
from utils import set_seed, make_clinical_reward_columns, DiffusionUtils, DiffusionValidator


if __name__ == "__main__":
    # ============================================================
    # CONFIG
    # ============================================================
    set_seed(42)

    interval = 4
    analysis_window_hours = 96
    encoder_epochs = 20
    batch_size = 256
    latent_dim = 7
    random_seed = 42

    terminal_reward_value = 5.0

    mc_samples = 30
    max_validation_transitions = 15000
    max_ood_train_samples = 50000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ============================================================
    # COHORT / RAW STATE DATA
    # ============================================================
    lab_join = """
        JOIN mimic.icustays i
        ON l.hadm_id = i.hadm_id
    """

    my_required_items = [
        {"item_id": 220045, "item_name": "BPM_inv"},
        {"item_id": 225309, "item_name": "BPM_noninv"},
        {"item_id": 220546, "item_name": "WBC"},
        {"item_id": 220179, "item_name": "NIBPs"},
        {"item_id": 220645, "item_name": "Sodium"},
        {"item_id": 220621, "item_name": "Glucose"},
        {"item_id": 220602, "item_name": "Chloride"},
        {"item_id": 220210, "item_name": "RR"},
        {"item_id": 224685, "item_name": "Tidal_Volume"},
        {"item_id": 220224, "item_name": "PaO2"},
        {"item_id": 223835, "item_name": "FiO2"},
        {"item_id": 227457, "item_name": "platelets_valuenum"},
        {"item_id": 227467, "item_name": "zinr"},
        {"item_id": 220228, "item_name": "HGB"},
        {"item_id": 227466, "item_name": "PTT"},
        {"item_id": 220545, "item_name": "Hematocrit"},
        {"item_id": 220615, "item_name": "creatinine_valuenum"},
        {"item_id": 225624, "item_name": "BUN"},
        {"item_id": 227443, "item_name": "bicarbonate"},
        {"item_id": 224828, "item_name": "Base_Excess_chart"},
        {"item_id": 225668, "item_name": "Lactate_chart"},
        {"item_id": 225690, "item_name": "tb_valuenum"},
        {"item_id": 220587, "item_name": "SGOT"},
        {"item_id": 227442, "item_name": "Potassium_chart"},
        {"item_id": 225667, "item_name": "Ionized_Calcium"},
        {"item_id": 223830, "item_name": "PH"},
        {"item_id": 225625, "item_name": "Calcium_non_ionized"},
        {
            "item_id": 51486, "item_name": "Lab_WBC",
            "table_name": "mimic_hosp.labevents l", "time_col": "l.charttime",
            "value_col": "l.valuenum", "stay_id_col": "i.stay_id",
            "join_clause": lab_join, "resample_method": "mean",
        },
        {
            "item_id": 51221, "item_name": "Lab_Hematocrit",
            "table_name": "mimic_hosp.labevents l", "time_col": "l.charttime",
            "value_col": "l.valuenum", "stay_id_col": "i.stay_id",
            "join_clause": lab_join, "resample_method": "mean",
        },
        {
            "item_id": 50802, "item_name": "ABE",
            "table_name": "mimic_hosp.labevents l", "time_col": "l.charttime",
            "value_col": "l.valuenum", "stay_id_col": "i.stay_id",
            "join_clause": lab_join, "resample_method": "mean",
        },
        {
            "item_id": 50813, "item_name": "Lactate",
            "table_name": "mimic_hosp.labevents l", "time_col": "l.charttime",
            "value_col": "l.valuenum", "stay_id_col": "i.stay_id",
            "join_clause": lab_join, "resample_method": "max",
        },
        {
            "item_id": 50971, "item_name": "Potassium",
            "table_name": "mimic_hosp.labevents l", "time_col": "l.charttime",
            "value_col": "l.valuenum", "stay_id_col": "i.stay_id",
            "join_clause": lab_join, "resample_method": "mean",
        },
    ]

    initial_values = {
        "NIBPs": 100, "NIBPd": 70, "heart_rate": 60, "SpO2": 90,
        "Temperature": 36, "Potassium": 3.5, "Glucose": 144, "Magnesium": 2.0,
        "SGOT": 5, "platelets_valuenum": 5, "zinr": 0.8, "P": 80, "ALT": 7,
        "Sodium": 135, "BUN": 7, "Calcium": 1.1, "tb_valuenum": 0.1,
        "PTT": 35, "PH": 7.35, "bicarbonate": 22, "RR": 12, "HGB": 7.0,
        "Chloride": 96, "creatinine_valuenum": 0.6, "PaCO2": 35,
        "WBC": 4500, "PT": 11, "pf_ratio": 400, "AL": 0.5, "F": 21,
    }

    zero_fill_cols = [
        "gcs_score", "ABE", "total_sofa_score",
        "vasopressor_eq", "SIRS", "shock_index",
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

    cohort = target_cohort(with_stay, from_stay, conn, cur)
    stayids = cohort.query()
    stay_str = ",".join(map(str, stayids))

    states = states_preprocessor(
        conn, cur, interval, my_required_items,
        stayids, initial_values, zero_fill_cols,
    )
    df_query = states.main()

    sofa_module = sofa(conn, cur, stayids)
    df_sofa = sofa_module.main()

    survival_module = survival_labeler(conn, cur, stayids)
    df_survival = survival_module.main()

    df_icu_time = pd.read_sql(
        f"""
        SELECT stay_id, intime, outtime
        FROM mimic.icustays
        WHERE stay_id IN ({stay_str})
        """,
        conn,
    )
    df_icu_time["intime"] = pd.to_datetime(df_icu_time["intime"])
    df_icu_time["outtime"] = pd.to_datetime(df_icu_time["outtime"])

    df_demo = pd.read_sql(
        f"""
        SELECT
            i.stay_id,
            p.anchor_age AS age,
            p.gender
        FROM mimic.icustays i
        JOIN mimic.patients p
        ON i.subject_id = p.subject_id
        WHERE i.stay_id IN ({stay_str})
        """,
        conn,
    )
    df_demo["age"] = pd.to_numeric(df_demo["age"], errors="coerce")
    df_demo["gender"] = df_demo["gender"].astype(str).str.upper()
    df_demo["sex_encoded"] = df_demo["gender"].map({"F": 0.0, "M": 1.0})
    df_demo["sex_encoded"] = df_demo["sex_encoded"].fillna(-1.0)

    df_gcs = pd.read_sql(
        f"""
        SELECT stay_id, time_hour AS charttime, gcs_score
        FROM hrl.gcs
        WHERE stay_id IN ({stay_str})
        """,
        conn,
    )
    df_gcs["charttime"] = pd.to_datetime(df_gcs["charttime"])

    # ============================================================
    # MERGE CLINICAL STATE / SOFA / SURVIVAL / DEMOGRAPHICS
    # ============================================================
    df_query["charttime"] = pd.to_datetime(df_query["charttime"])
    df_query = pd.merge(df_query, df_gcs, on=["stay_id", "charttime"], how="left")
    df_query["gcs_score"] = (
        df_query.groupby("stay_id")["gcs_score"]
        .ffill().bfill().fillna(15)
    )

    df_sofa = df_sofa.rename(
        columns={"chart_hour": "charttime", "total_sofa_score": "sofa_score"}
    )
    df_sofa["charttime"] = pd.to_datetime(df_sofa["charttime"])

    df_merged = pd.merge(
        df_query,
        df_sofa[["stay_id", "charttime", "sofa_score"]],
        on=["stay_id", "charttime"],
        how="left",
    )
    df_merged = pd.merge(df_merged, df_icu_time, on="stay_id", how="inner")
    df_merged = pd.merge(df_merged, df_demo, on="stay_id", how="left")

    df_merged = df_merged[
        (df_merged["charttime"] >= df_merged["intime"])
        & (df_merged["charttime"] <= df_merged["outtime"])
    ].copy()
    df_merged = df_merged.sort_values(["stay_id", "charttime"]).reset_index(drop=True)
    df_merged["sofa_score"] = (
        df_merged.groupby("stay_id")["sofa_score"]
        .ffill().bfill().fillna(0)
    )
    df_merged["age"] = df_merged["age"].fillna(
        df_merged["age"].median()
    )

    onset_mask = (df_merged["SIRS"] >= 2) | (df_merged["sofa_score"] >= 2)
    onset_df = (
        df_merged[onset_mask]
        .groupby("stay_id")["charttime"]
        .min()
        .reset_index()
        .rename(columns={"charttime": "onset_time"})
    )

    df_merged = pd.merge(df_merged, onset_df, on="stay_id", how="inner")
    df_merged["hours_from_onset"] = (
        (df_merged["charttime"] - df_merged["onset_time"])
        .dt.total_seconds() / 3600
    )

    if "Lactate_chart" in df_merged.columns and "Lactate" in df_merged.columns:
        df_merged["Lactate"] = df_merged["Lactate"].fillna(df_merged["Lactate_chart"])
        df_merged = df_merged.drop(columns=["Lactate_chart"])

    col_mapping = {
        "PaO2": "pao2",
        "FiO2": "fio2",
        "platelets_valuenum": "platelets",
        "tb_valuenum": "bilirubin",
        "creatinine_valuenum": "creatinine",
        "Lactate": "lactate",
        "gcs_score": "gcs",
    }
    df_merged = df_merged.rename(columns=col_mapping)

    df_merged = pd.merge(df_merged, df_survival, on="stay_id", how="left")
    df_merged = df_merged.dropna(subset=["survival"]).copy()
    df_merged["survival"] = df_merged["survival"].astype(int)

    df_merged["mortality"] = (df_merged["survival"] != 0).astype(int)

    # ============================================================
    # ACTION PREPROCESSING
    # ============================================================
    state_grid = df_merged[["stay_id", "charttime"]].drop_duplicates().copy()

    action_module = action_preprocessor(conn, cur, interval, stayids)
    df_actions = action_module.main(state_grid)

    if not df_actions.empty:
        df_actions["charttime"] = pd.to_datetime(df_actions["charttime"])
        df_full = pd.merge(
            df_merged, df_actions,
            on=["stay_id", "charttime"],
            how="left",
        )
        df_full["iv_fluid"] = df_full["iv_fluid"].fillna(0.0)
        df_full["vaso"] = df_full["vaso"].fillna(0.0)
        df_full["iv_action"] = df_full["iv_action"].fillna(1).astype(int)
        df_full["vaso_action"] = df_full["vaso_action"].fillna(1).astype(int)
        df_full["final_action"] = df_full["final_action"].fillna(0).astype(int)
    else:
        df_full = df_merged.copy()
        df_full["iv_fluid"] = 0.0
        df_full["vaso"] = 0.0
        df_full["iv_action"] = 1
        df_full["vaso_action"] = 1
        df_full["final_action"] = 0

    df_full = df_full.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    print("\n ICU Action distribution")
    print(df_full["final_action"].value_counts(normalize=True).sort_index())

    # ============================================================
    # ENCODER/HMM DATA PREPARATION
    # ============================================================
    df_window = df_full[
        df_full["hours_from_onset"].between(
            0, analysis_window_hours, inclusive="both"
        )
    ].copy()

    print(
        f"\nAnalysis window: onset 0-{analysis_window_hours}h"
        f" | stays={df_window['stay_id'].nunique()}"
        f" | rows={len(df_window)}"
    )

    sampler = Sample(
        data=df_query,
        iterations=1000,
        threshold=0.05,
        sample_size=1000,
        sofa=df_window,
    )
    df_sampled = sampler.main().rename(columns=col_mapping)
    df_sampled["charttime"] = pd.to_datetime(df_sampled["charttime"])

    if "onset_time" in df_sampled.columns:
        df_sampled = df_sampled.drop(columns=["onset_time"])

    df_sampled = pd.merge(df_sampled, onset_df, on="stay_id", how="inner")
    df_sampled["hours_from_onset"] = (
        (df_sampled["charttime"] - df_sampled["onset_time"])
        .dt.total_seconds() / 3600
    )
    df_sampled = df_sampled[
        df_sampled["hours_from_onset"].between(
            0, analysis_window_hours, inclusive="both"
        )
    ].copy()
    df_sampled = df_sampled.sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    sampled_stay_ids = df_sampled["stay_id"].unique().copy()

    dynamic_features = [
        "pao2", "fio2", "platelets", "bilirubin",
        "creatinine", "lactate", "gcs",
    ]
    next_dynamic_features = [f"{c}_next" for c in dynamic_features]

    static_context = ["age", "sex_encoded"]
    policy_state_description = ["latent_z"] + static_context

    df_sampled[dynamic_features] = (
        df_sampled.groupby("stay_id")[dynamic_features]
        .ffill().bfill().fillna(0)
    )

    # ============================================================
    # HMM + SDE ENCODER PRETRAINING
    # ============================================================
    hmm_module = SepsisHMM(n_components=4)
    hmm_module.train(df_sampled, dynamic_features)
    hmm_module.save_model()

    df_sampled["hmm_state"] = hmm_module.predict(
        df_sampled, dynamic_features
    )

    df_sampled[next_dynamic_features] = (
        df_sampled.groupby("stay_id")[dynamic_features].shift(-1)
    )
    df_sampled["encoder_next_charttime"] = (
        df_sampled.groupby("stay_id")["charttime"].shift(-1)
    )
    df_sampled["encoder_transition_hours"] = (
        (df_sampled["encoder_next_charttime"] - df_sampled["charttime"])
        .dt.total_seconds() / 3600
    )

    df_shifted = df_sampled.dropna(
        subset=next_dynamic_features + ["encoder_next_charttime"]
    ).copy()
    df_shifted = df_shifted[
        np.isclose(df_shifted["encoder_transition_hours"], interval)
    ].copy()
    df_shifted = df_shifted.sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    encoder_stay_ids = df_shifted["stay_id"].unique().copy()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(encoder_stay_ids)

    encoder_split = int(len(encoder_stay_ids) * 0.8)
    encoder_train_ids = encoder_stay_ids[:encoder_split]
    encoder_val_ids = encoder_stay_ids[encoder_split:]

    encoder_train_df = df_shifted[
        df_shifted["stay_id"].isin(encoder_train_ids)
    ].copy()
    encoder_val_df = df_shifted[
        df_shifted["stay_id"].isin(encoder_val_ids)
    ].copy()

    X_train = hmm_module.scaler.transform(
        encoder_train_df[dynamic_features].values
    )
    X_train_next = hmm_module.scaler.transform(
        encoder_train_df[next_dynamic_features].values
    )
    X_val = hmm_module.scaler.transform(
        encoder_val_df[dynamic_features].values
    )
    X_val_next = hmm_module.scaler.transform(
        encoder_val_df[next_dynamic_features].values
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_train_next),
        torch.LongTensor(encoder_train_df["hmm_state"].values),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(X_val_next),
        torch.LongTensor(encoder_val_df["hmm_state"].values),
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False,
    )

    encoder_module = SepsisEncoder(
        input_dim=len(dynamic_features),
        latent_dim=latent_dim,
        device=str(device),
    )
    encoder_module.train(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        epochs=encoder_epochs,
    )
    encoder_module.build_distributions(train_dataloader)
    encoder_module.save_model(path="sde_encoder_dict.pth")

    encoder_module.encoder.eval()
    encoder_module.sde.eval()
    encoder_module.decoder.eval()

    for param in encoder_module.encoder.parameters():
        param.requires_grad = False
    for param in encoder_module.sde.parameters():
        param.requires_grad = False

    # ============================================================
    # RL TRANSITION DATA PREPARATION
    # ============================================================
    df_rl = df_full[
        (~df_full["stay_id"].isin(sampled_stay_ids))
        & df_full["hours_from_onset"].between(
            0, analysis_window_hours, inclusive="both"
        )
    ].copy()
    df_rl = df_rl.sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    print(
        f"\nRL analysis window: onset 0-{analysis_window_hours}h"
        f" | stays={df_rl['stay_id'].nunique()}"
        f" | rows={len(df_rl)}"
    )

    overlap = set(sampled_stay_ids) & set(df_rl["stay_id"].unique())
    print("Encoder/RL overlap:", len(overlap))
    if overlap:
        raise ValueError("Encoder/RL overlap")

    df_rl[dynamic_features] = (
        df_rl.groupby("stay_id")[dynamic_features]
        .ffill().bfill().fillna(0)
    )
    df_rl[next_dynamic_features] = (
        df_rl.groupby("stay_id")[dynamic_features].shift(-1)
    )
    df_rl["sofa_score_next"] = (
        df_rl.groupby("stay_id")["sofa_score"].shift(-1)
    )
    df_rl["next_charttime"] = (
        df_rl.groupby("stay_id")["charttime"].shift(-1)
    )

    df_rl["action_to_next"] = (
        df_rl.groupby("stay_id")["final_action"].shift(-1)
    )
    df_rl["iv_fluid_to_next"] = (
        df_rl.groupby("stay_id")["iv_fluid"].shift(-1)
    )
    df_rl["vaso_to_next"] = (
        df_rl.groupby("stay_id")["vaso"].shift(-1)
    )

    df_rl["transition_hours"] = (
        (df_rl["next_charttime"] - df_rl["charttime"])
        .dt.total_seconds() / 3600
    )

    df_rl_shifted = df_rl.dropna(
        subset=next_dynamic_features
        + ["sofa_score_next", "next_charttime", "action_to_next"]
    ).copy()
    df_rl_shifted = df_rl_shifted[
        np.isclose(df_rl_shifted["transition_hours"], interval)
    ].copy()
    df_rl_shifted["action_to_next"] = (
        df_rl_shifted["action_to_next"].astype(int)
    )
    df_rl_shifted = df_rl_shifted.sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    df_rl_shifted["is_last"] = (
        df_rl_shifted.groupby("stay_id")
        .cumcount(ascending=False)
        .eq(0)
    )

    # ============================================================
    # CLINICAL REWARD PREPARATION
    # ============================================================
    df_rl_shifted = make_clinical_reward_columns(
        df_rl_shifted,
        terminal_reward_value=terminal_reward_value,
    )

    # ============================================================
    # TRAIN / VAL / TEST SPLIT
    # ============================================================
    unique_rl_stay_ids = df_rl_shifted["stay_id"].unique().copy()
    rng.shuffle(unique_rl_stay_ids)

    n_rl = len(unique_rl_stay_ids)
    train_end = int(n_rl * 0.70)
    val_end = int(n_rl * 0.85)

    train_stay_ids = unique_rl_stay_ids[:train_end]
    val_stay_ids = unique_rl_stay_ids[train_end:val_end]
    test_stay_ids = unique_rl_stay_ids[val_end:]

    df_rl_train = df_rl_shifted[
        df_rl_shifted["stay_id"].isin(train_stay_ids)
    ].copy()
    df_rl_val = df_rl_shifted[
        df_rl_shifted["stay_id"].isin(val_stay_ids)
    ].copy()
    df_rl_test = df_rl_shifted[
        df_rl_shifted["stay_id"].isin(test_stay_ids)
    ].copy()

    print("\nRL stays")
    print("Train:", len(train_stay_ids))
    print("Validation:", len(val_stay_ids))
    print("Test:", len(test_stay_ids))

    # ============================================================
    # DIFFUSION SCALE PREPARATION
    # ============================================================
    diffusion = DiffusionUtils(
        encoder_module=encoder_module,
        device=device,
    )

    X_rl_train = hmm_module.scaler.transform(
        df_rl_train[dynamic_features].values
    )
    diffusion_train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_rl_train)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    g_mean, g_std = diffusion.fit_stats(
        diffusion_train_loader
    )

    print("\nDiffusion raw mean:", g_mean)
    print("Diffusion raw std :", g_std)

    # ============================================================
    # DIFFUSION VALIDATION
    # ============================================================
    diffusion_validator = DiffusionValidator(
        project_root=PROJECT_ROOT,
        encoder_module=encoder_module,
        scaler=hmm_module.scaler,
        device=device,
        dynamic_features=dynamic_features,
        next_dynamic_features=next_dynamic_features,
        g_mean=g_mean,
        g_std=g_std,
        interval=interval,
        analysis_window_hours=analysis_window_hours,
        random_seed=random_seed,
    )

    diffusion_summary = diffusion_validator.run_all(
        df_train=df_rl_train,
        df_test=df_rl_test,
        mc_samples=mc_samples,
        max_validation_transitions=max_validation_transitions,
        max_ood_train_samples=max_ood_train_samples,
    )

    print("\nDiffusion validation summary")
    print(diffusion_summary.T)
