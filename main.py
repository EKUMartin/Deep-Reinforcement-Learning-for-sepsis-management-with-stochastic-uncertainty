from pathlib import Path
import sys
import torch
from Preprocessing.preprocessing import target_cohort,states_preprocessor,action_preprocessor,sofa
from Pretraining.encoder import SepsisEncoder
from torch.utils.data import TensorDataset, DataLoader
from Pretraining.hmm import SepsisHMM
from Pretraining.sampling import Sample
from Visualization.results import visualizer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from db_conn import db
import pandas as pd

# main.py
if __name__ == "__main__":
    #======================================================================
    # Initial Values & Queries
    #======================================================================
    interval = 4 # resampling hour interval
    lab_join = 'JOIN mimic.icustays i ON l.hadm_id = i.hadm_id'
    my_required_items = [
        {'item_id': 220045, 'item_name': 'BPM_inv'}, {'item_id': 225309, 'item_name': 'BPM_noninv'},
        {'item_id': 220546, 'item_name': 'WBC'}, {'item_id': 220179, 'item_name': 'NIBPs'},
        {'item_id': 220645, 'item_name': 'Sodium'}, {'item_id': 220621, 'item_name': 'Glucose'},
        {'item_id': 220602, 'item_name': 'Chloride'}, {'item_id': 220210, 'item_name': 'RR'},
        {'item_id': 224685, 'item_name': 'Tidal_Volume'}, {'item_id': 220224, 'item_name': 'PaO2'},
        {'item_id': 223835, 'item_name': 'FiO2'}, {'item_id': 227457, 'item_name': 'platelets_valuenum'},
        {'item_id': 227467, 'item_name': 'zinr'}, {'item_id': 220228, 'item_name': 'HGB'},
        {'item_id': 227466, 'item_name': 'PTT'}, {'item_id': 220545, 'item_name': 'Hematocrit'},
        {'item_id': 220615, 'item_name': 'creatinine_valuenum'}, {'item_id': 225624, 'item_name': 'BUN'},
        {'item_id': 227443, 'item_name': 'bicarbonate'}, {'item_id': 224828, 'item_name': 'Base_Excess_chart'},
        {'item_id': 225668, 'item_name': 'Lactate_chart'}, {'item_id': 225690, 'item_name': 'tb_valuenum'},
        {'item_id': 220587, 'item_name': 'SGOT'}, {'item_id': 227442, 'item_name': 'Potassium_chart'},
        {'item_id': 225667, 'item_name': 'Ionized_Calcium'}, {'item_id': 223830, 'item_name': 'PH'},
        {'item_id': 225625, 'item_name': 'Calcium_non_ionized'},
        {'item_id': 51486, 'item_name': 'Lab_WBC', 'table_name': 'mimic_hosp.labevents l', 'time_col': 'l.charttime', 'value_col': 'l.valuenum', 'stay_id_col': 'i.stay_id', 'join_clause': lab_join, 'resample_method': 'mean'},
        {'item_id': 51221, 'item_name': 'Lab_Hematocrit', 'table_name': 'mimic_hosp.labevents l', 'time_col': 'l.charttime', 'value_col': 'l.valuenum', 'stay_id_col': 'i.stay_id', 'join_clause': lab_join, 'resample_method': 'mean'},
        {'item_id': 50802, 'item_name': 'ABE', 'table_name': 'mimic_hosp.labevents l', 'time_col': 'l.charttime', 'value_col': 'l.valuenum', 'stay_id_col': 'i.stay_id', 'join_clause': lab_join, 'resample_method': 'mean'},
        {'item_id': 50813, 'item_name': 'Lactate', 'table_name': 'mimic_hosp.labevents l', 'time_col': 'l.charttime', 'value_col': 'l.valuenum', 'stay_id_col': 'i.stay_id', 'join_clause': lab_join, 'resample_method': 'max'},
        {'item_id': 50971, 'item_name': 'Potassium', 'table_name': 'mimic_hosp.labevents l', 'time_col': 'l.charttime', 'value_col': 'l.valuenum', 'stay_id_col': 'i.stay_id', 'join_clause': lab_join, 'resample_method': 'mean'}
    ]
    initial_values = {
        'NIBPs': 100, 'NIBPd': 70, "heart_rate": 60, 'SpO2': 90, 'Temperature': 36, 'Potassium': 3.5, 'Glucose': 144, 
        'Magnesium': 2.0, 'SGOT': 5, 'platelets_valuenum': 5, 'zinr': 0.8, 'P': 80, 'ALT': 7, 'Sodium': 135, 'BUN': 7, 
        'Calcium': 1.1, 'tb_valuenum': 0.1, 'PTT': 35, 'PH': 7.35, 'bicarbonate': 22, 'RR': 12, 'HGB': 7.0, 'Chloride': 96,
        'creatinine_valuenum': 0.6, 'PaCO2': 35, 'WBC': 4500, 'PT': 11, 'pf_ratio': 400, 'AL': 0.5, 'F': 21
    }
    zero_fill_cols = ['gcs_score', 'ABE', "total_sofa_score", "vasopressor_eq", "SIRS", "shock_index"]

    with_stay = """
        WITH ranked_stays AS (
            SELECT i.subject_id, i.stay_id, i.intime, i.first_careunit, p.anchor_age,
                ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime ASC) as rn
            FROM mimic.icustays i JOIN mimic.patients p ON i.subject_id = p.subject_id
        )
    """
    from_stay = """
            FROM ranked_stays WHERE rn = 1 AND anchor_age >= 18 AND stay_id IN (SELECT stay_id FROM hrl.sepsis3)
            AND first_careunit IN ('Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU)', 'Medical/Surgical Intensive Care Unit (MICU/SICU)', 'Intensive Care Unit (ICU)'); 
    """

    #======================================================================
    # DB connection and fetch data
    #======================================================================
    conn, cur = db.open_db()
    cohort = target_cohort(with_stay, from_stay, conn, cur)
    stayids = cohort.query()
    states = states_preprocessor(conn, cur, interval, my_required_items, stayids, initial_values, zero_fill_cols)
    df_query = states.main()
    sofa_ = sofa(conn, cur, stayids)
    df_sofa = sofa_.main()

    # Adding GCS
    stay_str = ','.join(map(str, stayids))
    q_gcs = f"SELECT stay_id, time_hour AS charttime, gcs_score FROM hrl.gcs WHERE stay_id IN ({stay_str})"
    df_gcs = pd.read_sql(q_gcs, conn)

    # Merging Results
    df_gcs['charttime'] = pd.to_datetime(df_gcs['charttime'])
    df_query['charttime'] = pd.to_datetime(df_query['charttime'])
    df_query = pd.merge(df_query, df_gcs, on=['stay_id', 'charttime'], how='left')
    df_query['gcs_score'] = df_query.groupby('stay_id')['gcs_score'].ffill().fillna(15)
    
    df_sofa = df_sofa.rename(columns={'chart_hour': 'charttime', 'total_sofa_score': 'sofa_score'})
    df_sofa['charttime'] = pd.to_datetime(df_sofa['charttime'])
    df_merged = pd.merge(df_query, df_sofa[['stay_id', 'charttime', 'sofa_score']], on=['stay_id', 'charttime'], how='left')
    df_merged['sofa_score'] = df_merged.groupby('stay_id')['sofa_score'].ffill().fillna(0)

    # Sepsis On-Set time calculation 
    onset_mask = (df_merged['SIRS'] >= 2) | (df_merged['sofa_score'] >= 2)
    onset_df = df_merged[onset_mask].groupby('stay_id')['charttime'].min().reset_index()
    onset_df = onset_df.rename(columns={'charttime': 'onset_time'})
    df_merged = pd.merge(df_merged, onset_df, on='stay_id', how='inner')
    df_merged['hours_from_onset'] = (df_merged['charttime'] - df_merged['onset_time']).dt.total_seconds() / 3600
    df_24h = df_merged[(df_merged['hours_from_onset'] >= 0) & (df_merged['hours_from_onset'] <= 24)].copy()
    
    if 'Lactate_chart' in df_query.columns and 'Lactate' in df_query.columns:
        df_query['Lactate'] = df_query['Lactate'].fillna(df_query['Lactate_chart'])
        df_query = df_query.drop(columns=['Lactate_chart'])

    # Data for HMM (24HRs from sepsis onset)
    col_mapping = {'PaO2': 'pao2', 'FiO2': 'fio2', 'platelets_valuenum': 'platelets', 'tb_valuenum': 'bilirubin', 'creatinine_valuenum': 'creatinine', 'Lactate': 'lactate', 'gcs_score': 'gcs'}
    df_24h = df_24h.rename(columns=col_mapping)

    #======================================================================
    # Action Preprocessing
    #======================================================================



    #======================================================================
    # sampling 1000 stayids
    #======================================================================
    sampler = Sample(data=df_query, iterations=1000, threshold=0.05, sample_size=1000, sofa=df_24h)
    df_sampled = sampler.main()
    df_sampled = df_sampled.rename(columns=col_mapping)

    # Used Variables
    features_col = ['pao2', 'fio2', 'platelets', 'bilirubin', 'creatinine', 'lactate', 'gcs']
    df_sampled[features_col] = df_sampled.groupby('stay_id')[features_col].ffill().bfill().fillna(0)

    # HMM Training
    hmm_module = SepsisHMM(n_components=4)
    hmm_module.train(df_sampled, features_col)
    hmm_module.save_model()

    # HMM result
    df_sampled['hmm_state'] = hmm_module.predict(df_sampled, features_col)
    df_sampled = pd.merge(df_sampled, df_sofa[['stay_id', 'charttime', 'sofa_score']], on=['stay_id', 'charttime'], how='left')
    stats_table = visualizer.get_hmm_state_stats(df_sampled)
    print(stats_table)

    #======================================================================
    #  building t+1 dataset for Encoder Training
    #======================================================================
    next_features_cols = [f"{c}_next" for c in features_col]
    df_sampled[next_features_cols] = df_sampled.groupby('stay_id')[features_col].shift(-1)
    df_shifted = df_sampled.dropna(subset=next_features_cols).copy()

    X_curr = hmm_module.scaler.transform(df_shifted[features_col].values)
    X_next = hmm_module.scaler.transform(df_shifted[next_features_cols].values)
    hmm_states = df_shifted['hmm_state'].values

    tensor_X = torch.FloatTensor(X_curr)
    tensor_X_next = torch.FloatTensor(X_next)
    tensor_state = torch.LongTensor(hmm_states)
    dataset = TensorDataset(tensor_X, tensor_X_next, tensor_state)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    full_dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Encoder Training
    encoder_module = SepsisEncoder(input_dim=len(features_col), latent_dim=8, device='cuda')
    

    encoder_module.train(train_loader=train_dataloader, val_loader=val_dataloader, epochs=20)
    
    encoder_module.build_distributions(train_dataloader)
    encoder_module.evaluate_test_set(val_dataloader)
    
    encoder_module.save_model(path='sde_encoder_dict.pth')
    
    # Encoder Training Result Visualization
    visualizer.visualize_latent_tsne_full(encoder_module, full_dataloader)
    df_result_norms_stage = visualizer.extract_and_visualize_fg_norms_by_stage(encoder_module, full_dataloader)
    visualizer.visualize_sde_trajectory_pca(encoder_module, val_dataloader, num_samples=3)

    #======================================================================
    # Training 
    #======================================================================
    



    #======================================================================
    # Training Result
    #======================================================================