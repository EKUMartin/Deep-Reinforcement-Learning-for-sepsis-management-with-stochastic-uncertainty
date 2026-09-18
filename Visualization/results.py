import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchsde
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
class visualizer:
    def __init__(self):
        pass
    def visualize_latent_tsne_full(self,sepsis_encoder, data_loader): # TSNE results mu of encoder training
        print("\nExtracting latent representations for t-SNE...")
        sepsis_encoder.encoder.eval()
        mu_list, labels_list = [], []
        
        with torch.no_grad():
            for batch_X, _, batch_s in data_loader:
                batch_X = batch_X.to(sepsis_encoder.device)
                mu, _ = sepsis_encoder.encoder(batch_X)
                mu_list.append(mu.cpu().numpy())
                labels_list.append(batch_s.numpy())
                
        mu_all = np.concatenate(mu_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)
            
        print(f"Running t-SNE on {len(mu_all)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30.0, max_iter=1000)
        tsne_results = tsne.fit_transform(mu_all)

        plt.figure(figsize=(10, 8))
        num_classes = len(np.unique(labels_all))
        palette = sns.color_palette("Set2", num_classes)
        
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels_all, palette=palette, alpha=0.7, s=30, edgecolor=None)
        plt.title('t-SNE Visualization of Sepsis States', fontsize=14)
        plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Sepsis State', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(); plt.show()

    def get_hmm_state_stats(self,df_hmm): # get HMM states
        results = []
        for state_id, group in df_hmm.groupby('hmm_state'):
            res = {'HMM State': f"State {state_id}"}
            n_obs = len(group)
            res['Total Observations'] = f"{n_obs} ({n_obs/len(df_hmm)*100:.1f}%)"
            res['Unique Patients'] = str(group['stay_id'].nunique())
            
            def get_iqr(series):
                s = pd.to_numeric(series, errors='coerce').dropna()
                if len(s) == 0: return "N/A"
                return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"
            
            res['SOFA Score, Median(IQR)'] = get_iqr(group.get('sofa_score', pd.Series([])))
            res['PaO2, Median(IQR)'] = get_iqr(group['pao2'])
            res['FiO2, Median(IQR)'] = get_iqr(group['fio2'])
            res['Platelets, Median(IQR)'] = get_iqr(group['platelets'])
            res['Bilirubin, Median(IQR)'] = get_iqr(group['bilirubin'])
            res['Creatinine, Median(IQR)'] = get_iqr(group['creatinine'])
            res['Lactate, Median(IQR)'] = get_iqr(group.get('lactate', pd.Series([])))
            res['GCS, Median(IQR)'] = get_iqr(group['gcs'])
            results.append(res)
        return pd.DataFrame(results).set_index('HMM State').T

    def extract_and_visualize_fg_norms_by_stage(self,encoder_module, data_loader): # f,g L2 Norm & LMax  Norm Results
        print("\nExtracting pure Drift (f) and Diffusion (g) vectors by Stage...")
        encoder_module.encoder.eval(); encoder_module.sde.eval()
        f_l2_list, f_max_list, g_l2_list, g_max_list, stage_list = [], [], [], [], []
        device = encoder_module.device
        
        with torch.no_grad():
            for batch_X, _, batch_s in data_loader:
                batch_X = batch_X.to(device)
                mu, _ = encoder_module.encoder(batch_X)
                t_zero = torch.zeros(mu.shape[0], 1, device=device)
                ty = torch.cat([t_zero, mu], dim=-1)
                
                f_val = encoder_module.sde.f_net(ty)
                g_val = encoder_module.sde.g_net(ty)
                
                f_l2_list.extend(torch.norm(f_val, p=2, dim=-1).cpu().numpy())
                f_max_list.extend(torch.norm(f_val, p=float('inf'), dim=-1).cpu().numpy())
                g_l2_list.extend(torch.norm(g_val, p=2, dim=-1).cpu().numpy())
                g_max_list.extend(torch.norm(g_val, p=float('inf'), dim=-1).cpu().numpy())
                stage_list.extend(batch_s.cpu().numpy())

        df_norms = pd.DataFrame({'Stage': stage_list, 'F_L2_Norm': f_l2_list, 'F_Max_Norm': f_max_list, 'G_L2_Norm': g_l2_list, 'G_Max_Norm': g_max_list})
        print("\n--- Descriptive Statistics by Stage ---")
        print(df_norms.groupby('Stage').describe().T)
        
        print("\nPlotting distributions by Stage...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.boxplot(data=df_norms, x='Stage', y='F_L2_Norm', ax=axes[0, 0], palette="Set2"); axes[0, 0].set_title('Drift (f) - L2 Norm')
        sns.boxplot(data=df_norms, x='Stage', y='F_Max_Norm', ax=axes[0, 1], palette="Set2"); axes[0, 1].set_title('Drift (f) - Max Norm')
        sns.boxplot(data=df_norms, x='Stage', y='G_L2_Norm', ax=axes[1, 0], palette="Set2"); axes[1, 0].set_title('Diffusion (g) - L2 Norm')
        sns.boxplot(data=df_norms, x='Stage', y='G_Max_Norm', ax=axes[1, 1], palette="Set2"); axes[1, 1].set_title('Diffusion (g) - Max Norm')
        plt.tight_layout(); plt.show()
        return df_norms

    def visualize_sde_trajectory_pca(self,encoder_module, data_loader, num_samples=5, time_steps=50): # Trajectory reconstruction from SDE solver
        print("\nGenerating SDE continuous trajectories ($z_t \\rightarrow z_{t+1}$)...")
        encoder_module.encoder.eval(); encoder_module.sde.eval()
        device = encoder_module.device
        
        batch_X, _, batch_s = next(iter(data_loader))
        batch_X = batch_X[:num_samples].to(device)
        stages = batch_s[:num_samples].numpy()
        
        with torch.no_grad():
            mu, _ = encoder_module.encoder(batch_X)
            ts = torch.linspace(0.0, 1.0, steps=time_steps, device=device)
            y0_aug = torch.cat([mu, torch.zeros(mu.shape[0], 1, device=device)], dim=-1)
            y_aug_ts = torchsde.sdeint(encoder_module.sde, y0_aug, ts, method='euler', dt=0.05)
            z_traj = y_aug_ts[:, :, :-1].cpu().numpy()

        z_traj_flat = z_traj.reshape(-1, z_traj.shape[-1])
        pca = PCA(n_components=2)
        z_traj_2d = pca.fit_transform(z_traj_flat).reshape(time_steps, num_samples, 2)

        plt.figure(figsize=(12, 10))
        colors = plt.get_cmap('Set1', num_samples)

        for i in range(num_samples):
            traj = z_traj_2d[:, i, :]
            plt.plot(traj[:, 0], traj[:, 1], color=colors(i), alpha=0.6, linewidth=2.5, label=f'Patient {i+1} (Stage {stages[i]})')
            plt.scatter(traj[0, 0], traj[0, 1], color=colors(i), marker='o', s=150, edgecolors='black', zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], color=colors(i), marker='X', s=200, edgecolors='black', zorder=5)

        plt.title('2D PCA Projection of Continuous SDE Trajectories ($z_t \\rightarrow z_{t+1}$)', fontsize=15)
        plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(); plt.show()