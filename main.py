from pathlib import Path
import sys
import torch
from Preprocessing.preprocessing import target_cohort,states_preprocessor,action_preprocessor,sofa
from Pretraining.encoder import SepsisEncoder
from torch.utils.data import TensorDataset, DataLoader
from Pretraining.hmm import SepsisHMM
from Pretraining.sampling import Sample
from Visualization.results import visualizer
from policy import LowLevelQNetwork, HighLevelPolicy
import torch.optim as optim
import numpy as np
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
    rl_epochs = 30
    gamma = 0.99      
    lambda_pt = 2.25   

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
    action_module = action_preprocessor(conn, cur, interval, stayids)
    df_actions = action_module.main()
    if not df_actions.empty:
        df_actions['charttime'] = df_actions['time_bin'].apply(lambda x: x.right if pd.notnull(x) else pd.NaT)
        df_actions['charttime'] = pd.to_datetime(df_actions['charttime'])
        action_cols = ['stay_id', 'charttime', 'iv_fluid', 'vaso', 'iv_action', 'vaso_action', 'final_action']
        df_actions = df_actions[action_cols]
        df_24h = pd.merge(df_24h, df_actions, on=['stay_id', 'charttime'], how='left')
        df_24h['iv_fluid'] = df_24h['iv_fluid'].fillna(0)
        df_24h['vaso'] = df_24h['vaso'].fillna(0)
        df_24h['iv_action'] = df_24h['iv_action'].fillna(1).astype(int)
        df_24h['vaso_action'] = df_24h['vaso_action'].fillna(1).astype(int)
        df_24h['final_action'] = df_24h['final_action'].fillna(0).astype(int)
    else:
        print("Action data is empty.")

    #======================================================================
    # sampling 1000 stayids
    #======================================================================
    sampler = Sample(data=df_query, iterations=1000, threshold=0.05, sample_size=1000, sofa=df_24h)
    df_sampled = sampler.main()
    df_sampled = df_sampled.rename(columns=col_mapping)
    sampled_stay_ids = df_sampled['stay_id'].unique()
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
    encoder_module = SepsisEncoder(input_dim=len(features_col), latent_dim=7, device='cuda')
    

    encoder_module.train(train_loader=train_dataloader, val_loader=val_dataloader, epochs=20)
    
    encoder_module.build_distributions(train_dataloader)
    encoder_module.evaluate_test_set(val_dataloader)
    
    encoder_module.save_model(path='sde_encoder_dict.pth')
    
    # Encoder Training Result Visualization
    visualizer.visualize_latent_tsne_full(encoder_module, full_dataloader)
    df_result_norms_stage = visualizer.extract_and_visualize_fg_norms_by_stage(encoder_module, full_dataloader)
    visualizer.visualize_sde_trajectory_pca(encoder_module, val_dataloader, num_samples=3)

    #======================================================================
    #  building dataset for RL Training
    #======================================================================
    
    df_rl = df_24h[~df_24h['stay_id'].isin(sampled_stay_ids)].copy()
    df_rl[features_col] = df_rl.groupby('stay_id')[features_col].ffill().bfill().fillna(0)

    df_rl[next_features_cols] = df_rl.groupby('stay_id')[features_col].shift(-1)
    df_rl['sofa_score_next'] = df_rl.groupby('stay_id')['sofa_score'].shift(-1)

    df_rl_shifted = df_rl.dropna(subset=next_features_cols + ['sofa_score_next']).copy()

    unique_rl_stay_ids = df_rl_shifted['stay_id'].unique()
    np.random.shuffle(unique_rl_stay_ids)
    split_idx = int(0.8 * len(unique_rl_stay_ids))
    
    train_stay_ids = unique_rl_stay_ids[:split_idx]
    test_stay_ids = unique_rl_stay_ids[split_idx:]

    df_rl_train = df_rl_shifted[df_rl_shifted['stay_id'].isin(train_stay_ids)]
    df_rl_test = df_rl_shifted[df_rl_shifted['stay_id'].isin(test_stay_ids)]

    X_train_rl = hmm_module.scaler.transform(df_rl_train[features_col].values)
    X_next_train_rl = hmm_module.scaler.transform(df_rl_train[next_features_cols].values)
    rl_train_dataset = TensorDataset(
        torch.FloatTensor(X_train_rl), torch.FloatTensor(X_next_train_rl),
        torch.LongTensor(df_rl_train['final_action'].values),
        torch.FloatTensor(df_rl_train['sofa_score'].values),
        torch.FloatTensor(df_rl_train['sofa_score_next'].values)
    )
    rl_train_dataloader = DataLoader(rl_train_dataset, batch_size=256, shuffle=True, drop_last=True)

    X_test_rl = hmm_module.scaler.transform(df_rl_test[features_col].values)
    X_next_test_rl = hmm_module.scaler.transform(df_rl_test[next_features_cols].values)
    rl_test_dataset = TensorDataset(
        torch.FloatTensor(X_test_rl), torch.FloatTensor(X_next_test_rl),
        torch.LongTensor(df_rl_test['final_action'].values),
        torch.FloatTensor(df_rl_test['sofa_score'].values),
        torch.FloatTensor(df_rl_test['sofa_score_next'].values)
    )
    rl_test_dataloader = DataLoader(rl_test_dataset, batch_size=256, shuffle=False)
    
    print(f"RL Training Dataset: {len(train_stay_ids)} patients, {len(rl_train_dataset)} transitions")
    print(f"RL Test Dataset: {len(test_stay_ids)} patients, {len(rl_test_dataset)} transitions")
    #======================================================================
    # Training 
    #======================================================================
    print("\nInitializing Hierarchical RL Agents...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    q_net = LowLevelQNetwork(latent_dim=7, action_dim=25).to(device)
    p_net = LowLevelQNetwork(latent_dim=7, action_dim=25).to(device)
    meta_net = HighLevelPolicy(latent_dim=7).to(device)
    
    opt_q = optim.Adam(q_net.parameters(), lr=1e-3)
    opt_p = optim.Adam(p_net.parameters(), lr=1e-3)
    opt_meta = optim.Adam(meta_net.parameters(), lr=5e-4)

    encoder_module.encoder.eval()
    encoder_module.sde.eval()
    for epoch in range(rl_epochs):
        q_net.train(); p_net.train(); meta_net.train()
        total_q_loss, total_p_loss, total_meta_loss = 0, 0, 0
        
        for batch_X, batch_X_next, batch_act, batch_sofa, batch_sofa_next in rl_train_dataloader:
            batch_X = batch_X.to(device)            
            batch_X_next = batch_X_next.to(device)
            batch_act = batch_act.to(device)
            batch_sofa, batch_sofa_next = batch_sofa.to(device), batch_sofa_next.to(device)
            
            with torch.no_grad(): 
                z_curr, _ = encoder_module.encoder(batch_X)
                z_next, _ = encoder_module.encoder(batch_X_next)
                
                ty = torch.cat([torch.zeros_like(z_curr[:, :1]), z_curr], dim=-1)
                g_val = encoder_module.sde.g_net(ty)
                g_norm = torch.norm(g_val, dim=1)
                
                g_norm_scaled = torch.clamp((g_norm - g_norm.mean()) / (g_norm.std() + 1e-5), 0, 1)

            reward_q = batch_sofa - batch_sofa_next
            
            reward_p = torch.where(reward_q >= 0, reward_q, reward_q * lambda_pt)

            curr_q = q_net(z_curr).gather(1, batch_act.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = q_net(z_next).max(1)[0]
            target_q = reward_q + gamma * next_q
            loss_q = torch.nn.functional.mse_loss(curr_q, target_q)
            
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()

            curr_p = p_net(z_curr).gather(1, batch_act.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_p = p_net(z_next).max(1)[0]
            target_p = reward_p + gamma * next_p
            loss_p = torch.nn.functional.mse_loss(curr_p, target_p)
            
            opt_p.zero_grad()
            loss_p.backward()
            opt_p.step()

            action_meta, log_prob, probs = meta_net.get_action(z_curr, g_norm_scaled)

            meta_reward = torch.where(action_meta == 0, reward_q, reward_p)
            baseline = meta_reward.mean()
            
            loss_meta, p_loss, e_loss, g_loss = meta_net.compute_loss(
                log_prob, probs, meta_reward, g_norm_scaled, baseline_reward=baseline, alpha=0.5, beta=0.01
            )
            
            opt_meta.zero_grad()
            loss_meta.backward()
            opt_meta.step()
            
            total_q_loss += loss_q.item()
            total_p_loss += loss_p.item()
            total_meta_loss += loss_meta.item()
            
        n_batches = len(train_dataloader)
        print(f"RL Epoch {epoch+1:02d} | Q_Loss: {total_q_loss/n_batches:.4f} | P_Loss: {total_p_loss/n_batches:.4f} | Meta_Loss: {total_meta_loss/n_batches:.4f}")

    torch.save({
        'q_net': q_net.state_dict(),
        'p_net': p_net.state_dict(),
        'meta_net': meta_net.state_dict()
    }, 'hrl_agents_dict.pth')

    #======================================================================
    # Training Result
    #======================================================================