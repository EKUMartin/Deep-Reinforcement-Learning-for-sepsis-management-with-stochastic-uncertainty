from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchsde

from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class BehaviorPolicyNetwork(nn.Module):
    def __init__(
        self,
        latent_dim=7,
        action_dim=25
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            latent_dim,
            128
        )

        self.fc2 = nn.Linear(
            128,
            128
        )

        self.out = nn.Linear(
            128,
            action_dim
        )

    def forward(self, z):
        x = F.relu(
            self.fc1(z)
        )

        x = F.relu(
            self.fc2(x)
        )

        return self.out(x)


class visualizer:
    def __init__(
        self,
        output_dir="results"
    ):
        self.output_dir = Path(
            output_dir
        )

        self.eval_dir = (
            self.output_dir
            / "evaluation"
        )

        self.sde_dir = (
            self.output_dir
            / "sde"
        )

        self.hmm_dir = (
            self.output_dir
            / "hmm"
        )

        self.hyper_dir = (
            self.output_dir
            / "hyperparameter"
        )

        self.eval_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.sde_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.hmm_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.hyper_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _device(
        self,
        encoder_module
    ):
        return torch.device(
            str(
                encoder_module.device
            )
        )

    def _get_latent(
        self,
        encoder_module,
        x
    ):
        mu, _ = (
            encoder_module
            .encoder(x)
        )

        return mu

    def _get_uncertainty(
        self,
        encoder_module,
        z
    ):
        t = torch.zeros_like(
            z[:, :1]
        )

        ty = torch.cat(
            [t, z],
            dim=1
        )

        g = (
            encoder_module
            .sde
            .g_net(ty)
            + 1e-3
        )

        return torch.max(
            g,
            dim=1
        ).values

    def _scale_uncertainty(
        self,
        g,
        g_mean,
        g_std
    ):
        return torch.sigmoid(
            (
                g - g_mean
            )
            /
            (
                g_std + 1e-6
            )
        )

    def _make_reward(
        self,
        df,
        terminal_reward=5.0
    ):
        step_reward = (
            df[
                'sofa_score'
            ].to_numpy(
                dtype=float
            )
            -
            df[
                'sofa_score_next'
            ].to_numpy(
                dtype=float
            )
        )

        step_reward = (
            np.clip(
                step_reward,
                -4.0,
                4.0
            )
            / 4.0
        )

        terminal = np.where(
            df[
                'survival'
            ].to_numpy()
            == 0,
            terminal_reward,
            -terminal_reward
        )

        is_last = (
            df[
                'is_last'
            ]
            .astype(float)
            .to_numpy()
        )

        return (
            step_reward
            + is_last
            * terminal
        )

    def _save_bar(
        self,
        labels,
        values,
        title,
        ylabel,
        path
    ):
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        x = np.arange(
            len(labels)
        )

        bars = ax.bar(
            x,
            values
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

        for bar, value in zip(
            bars,
            values
        ):
            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha='center',
                va='bottom'
            )

        fig.tight_layout()

        fig.savefig(
            path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

    def extract_policy_results(
        self,
        df_eval,
        scaler,
        features_col,
        encoder_module,
        q_net,
        p_net,
        high_net,
        g_mean,
        g_std,
        batch_size=4096
    ):
        device = self._device(
            encoder_module
        )

        q_net.eval()
        p_net.eval()
        high_net.eval()

        encoder_module.encoder.eval()
        encoder_module.sde.eval()

        df = (
            df_eval
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

        X = scaler.transform(
            df[
                features_col
            ].values
        )

        q_action_list = []
        p_action_list = []
        high_option_list = []
        final_action_list = []

        q_value_list = []
        p_value_list = []
        high_value_list = []

        uncertainty_list = []

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

                x = torch.FloatTensor(
                    X[
                        start:end
                    ]
                ).to(device)

                z = self._get_latent(
                    encoder_module,
                    x
                )

                g = self._get_uncertainty(
                    encoder_module,
                    z
                )

                g_scaled = (
                    self
                    ._scale_uncertainty(
                        g,
                        g_mean,
                        g_std
                    )
                )

                q_values = q_net(
                    z
                )

                p_values = p_net(
                    z
                )

                high_values = high_net(
                    z,
                    g_scaled
                )

                q_action = (
                    q_values
                    .argmax(
                        dim=1
                    )
                )

                p_action = (
                    p_values
                    .argmax(
                        dim=1
                    )
                )

                high_option = (
                    high_values
                    .argmax(
                        dim=1
                    )
                )

                final_action = (
                    torch.where(
                        high_option == 0,
                        q_action,
                        p_action
                    )
                )

                q_value = (
                    q_values
                    .max(
                        dim=1
                    )
                    .values
                )

                p_value = (
                    p_values
                    .max(
                        dim=1
                    )
                    .values
                )

                high_value = (
                    high_values
                    .max(
                        dim=1
                    )
                    .values
                )

                q_action_list.append(
                    q_action
                    .cpu()
                    .numpy()
                )

                p_action_list.append(
                    p_action
                    .cpu()
                    .numpy()
                )

                high_option_list.append(
                    high_option
                    .cpu()
                    .numpy()
                )

                final_action_list.append(
                    final_action
                    .cpu()
                    .numpy()
                )

                q_value_list.append(
                    q_value
                    .cpu()
                    .numpy()
                )

                p_value_list.append(
                    p_value
                    .cpu()
                    .numpy()
                )

                high_value_list.append(
                    high_value
                    .cpu()
                    .numpy()
                )

                uncertainty_list.append(
                    g_scaled
                    .cpu()
                    .numpy()
                )

        df[
            'clinician_action'
        ] = (
            df[
                'action_to_next'
            ]
            .astype(int)
        )

        df[
            'q_action'
        ] = np.concatenate(
            q_action_list
        )

        df[
            'prospect_action'
        ] = np.concatenate(
            p_action_list
        )

        df[
            'high_option'
        ] = np.concatenate(
            high_option_list
        )

        df[
            'hrl_action'
        ] = np.concatenate(
            final_action_list
        )

        df[
            'q_value'
        ] = np.concatenate(
            q_value_list
        )

        df[
            'prospect_value'
        ] = np.concatenate(
            p_value_list
        )

        df[
            'high_q_value'
        ] = np.concatenate(
            high_value_list
        )

        df[
            'uncertainty'
        ] = np.concatenate(
            uncertainty_list
        )

        df[
            'mortality'
        ] = (
            df[
                'survival'
            ]
            .ne(0)
            .astype(int)
        )

        df[
            'clinician_iv_level'
        ] = (
            df[
                'clinician_action'
            ]
            // 5
        )

        df[
            'clinician_vaso_level'
        ] = (
            df[
                'clinician_action'
            ]
            % 5
        )

        df[
            'hrl_iv_level'
        ] = (
            df[
                'hrl_action'
            ]
            // 5
        )

        df[
            'hrl_vaso_level'
        ] = (
            df[
                'hrl_action'
            ]
            % 5
        )

        df.to_csv(
            self.eval_dir
            / "policy_predictions.csv",
            index=False
        )

        return df

    def evaluate_vs_clinicians(
        self,
        policy_df
    ):
        rows = []

        policies = {
            'HRL':
                'hrl_action',

            'Only Q':
                'q_action',

            'Only Prospect':
                'prospect_action'
        }

        clinician = (
            policy_df[
                'clinician_action'
            ]
            .to_numpy()
        )

        nonzero_mask = (
            clinician != 0
        )

        for name, col in (
            policies.items()
        ):
            pred = (
                policy_df[
                    col
                ]
                .to_numpy()
            )

            agreement = (
                pred == clinician
            ).mean()

            if (
                nonzero_mask.sum()
                > 0
            ):
                nonzero_agreement = (
                    pred[
                        nonzero_mask
                    ]
                    ==
                    clinician[
                        nonzero_mask
                    ]
                ).mean()
            else:
                nonzero_agreement = (
                    np.nan
                )

            rows.append(
                {
                    'Policy':
                        name,

                    'Agreement':
                        agreement,

                    'Nonzero Agreement':
                        nonzero_agreement,

                    'Predicted Action0 Ratio':
                        (
                            pred == 0
                        ).mean(),

                    'Mean Absolute Action Difference':
                        np.abs(
                            pred
                            - clinician
                        ).mean()
                }
            )

        result = pd.DataFrame(
            rows
        )

        result.to_csv(
            self.eval_dir
            / "vs_clinicians.csv",
            index=False
        )

        x = np.arange(
            len(result)
        )

        width = 0.35

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.bar(
            x
            - width / 2,
            result[
                'Agreement'
            ],
            width,
            label='Overall'
        )

        ax.bar(
            x
            + width / 2,
            result[
                'Nonzero Agreement'
            ],
            width,
            label='Nonzero Actions'
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            result[
                'Policy'
            ]
        )

        ax.set_ylabel(
            'Agreement'
        )

        ax.set_title(
            'Policy vs Clinician Actions'
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            self.eval_dir
            / "vs_clinicians.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        return result

    def _encode_dataframe(
        self,
        df,
        scaler,
        features_col,
        encoder_module,
        batch_size=4096
    ):
        device = self._device(
            encoder_module
        )

        X = scaler.transform(
            df[
                features_col
            ].values
        )

        latent = []

        encoder_module.encoder.eval()

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

                x = torch.FloatTensor(
                    X[
                        start:end
                    ]
                ).to(device)

                z = self._get_latent(
                    encoder_module,
                    x
                )

                latent.append(
                    z.cpu()
                )

        return torch.cat(
            latent,
            dim=0
        )

    def train_behavior_policy(
        self,
        df_train,
        scaler,
        features_col,
        encoder_module,
        action_dim=25,
        epochs=5,
        batch_size=1024,
        lr=1e-3
    ):
        device = self._device(
            encoder_module
        )

        z_train = (
            self
            ._encode_dataframe(
                df_train,
                scaler,
                features_col,
                encoder_module
            )
        )

        actions = torch.LongTensor(
            df_train[
                'action_to_next'
            ]
            .astype(int)
            .values
        )

        dataset = TensorDataset(
            z_train,
            actions
        )

        loader = DataLoader(
            dataset,
            batch_size=
                batch_size,
            shuffle=True,
            drop_last=False
        )

        model = (
            BehaviorPolicyNetwork(
                latent_dim=
                    z_train.shape[1],
                action_dim=
                    action_dim
            )
            .to(device)
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        history = []

        for epoch in range(
            epochs
        ):
            model.train()

            total_loss = 0.0
            total_correct = 0
            total = 0

            for z, action in loader:

                z = z.to(device)
                action = action.to(
                    device
                )

                logits = model(z)

                loss = (
                    F.cross_entropy(
                        logits,
                        action
                    )
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                    * action.size(0)
                )

                total_correct += (
                    logits.argmax(
                        dim=1
                    )
                    == action
                ).sum().item()

                total += (
                    action.size(0)
                )

            history.append(
                {
                    'Epoch':
                        epoch + 1,

                    'Loss':
                        total_loss
                        / total,

                    'Accuracy':
                        total_correct
                        / total
                }
            )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            self.eval_dir
            / "behavior_policy_training.csv",
            index=False
        )

        torch.save(
            model.state_dict(),
            self.eval_dir
            / "behavior_policy.pth"
        )

        return model

    def _policy_probabilities(
        self,
        df,
        scaler,
        features_col,
        encoder_module,
        q_net,
        p_net,
        high_net,
        behavior_model,
        g_mean,
        g_std,
        temperature=1.0,
        batch_size=4096
    ):
        device = self._device(
            encoder_module
        )

        df = (
            df
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

        X = scaler.transform(
            df[
                features_col
            ].values
        )

        behavior_prob_list = []
        q_prob_list = []
        p_prob_list = []
        hrl_prob_list = []

        encoder_module.encoder.eval()
        encoder_module.sde.eval()

        q_net.eval()
        p_net.eval()
        high_net.eval()
        behavior_model.eval()

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

                x = torch.FloatTensor(
                    X[
                        start:end
                    ]
                ).to(device)

                z = self._get_latent(
                    encoder_module,
                    x
                )

                g = self._get_uncertainty(
                    encoder_module,
                    z
                )

                g_scaled = (
                    self
                    ._scale_uncertainty(
                        g,
                        g_mean,
                        g_std
                    )
                )

                q_values = q_net(z)

                p_values = p_net(z)

                high_values = high_net(
                    z,
                    g_scaled
                )

                q_probs = (
                    F.softmax(
                        q_values
                        / temperature,
                        dim=1
                    )
                )

                p_probs = (
                    F.softmax(
                        p_values
                        / temperature,
                        dim=1
                    )
                )

                high_probs = (
                    F.softmax(
                        high_values
                        / temperature,
                        dim=1
                    )
                )

                hrl_probs = (
                    high_probs[
                        :, 0:1
                    ]
                    * q_probs
                    +
                    high_probs[
                        :, 1:2
                    ]
                    * p_probs
                )

                behavior_probs = (
                    F.softmax(
                        behavior_model(
                            z
                        ),
                        dim=1
                    )
                )

                behavior_prob_list.append(
                    behavior_probs
                    .cpu()
                    .numpy()
                )

                q_prob_list.append(
                    q_probs
                    .cpu()
                    .numpy()
                )

                p_prob_list.append(
                    p_probs
                    .cpu()
                    .numpy()
                )

                hrl_prob_list.append(
                    hrl_probs
                    .cpu()
                    .numpy()
                )

        return (
            df,
            np.concatenate(
                behavior_prob_list
            ),
            np.concatenate(
                q_prob_list
            ),
            np.concatenate(
                p_prob_list
            ),
            np.concatenate(
                hrl_prob_list
            )
        )

    def evaluate_wis(
        self,
        df_train,
        df_test,
        scaler,
        features_col,
        encoder_module,
        q_net,
        p_net,
        high_net,
        g_mean,
        g_std,
        gamma=0.99,
        temperature=1.0,
        behavior_epochs=5,
        min_prob=1e-6
    ):
        behavior_model = (
            self
            .train_behavior_policy(
                df_train,
                scaler,
                features_col,
                encoder_module,
                epochs=
                    behavior_epochs
            )
        )

        (
            df,
            behavior_probs,
            q_probs,
            p_probs,
            hrl_probs
        ) = (
            self
            ._policy_probabilities(
                df_test,
                scaler,
                features_col,
                encoder_module,
                q_net,
                p_net,
                high_net,
                behavior_model,
                g_mean,
                g_std,
                temperature=
                    temperature
            )
        )

        actions = (
            df[
                'action_to_next'
            ]
            .astype(int)
            .to_numpy()
        )

        row_index = np.arange(
            len(df)
        )

        behavior_action_prob = (
            behavior_probs[
                row_index,
                actions
            ]
        )

        target_probs = {
            'HRL':
                hrl_probs[
                    row_index,
                    actions
                ],

            'Only Q':
                q_probs[
                    row_index,
                    actions
                ],

            'Only Prospect':
                p_probs[
                    row_index,
                    actions
                ]
        }

        behavior_action_prob = (
            np.clip(
                behavior_action_prob,
                min_prob,
                1.0
            )
        )

        reward = self._make_reward(
            df
        )

        base = pd.DataFrame(
            {
                'stay_id':
                    df[
                        'stay_id'
                    ].values,

                'reward':
                    reward,

                'behavior_prob':
                    behavior_action_prob
            }
        )

        summary_rows = []
        trajectory_rows = []

        observed_returns = []

        for stay_id, group in (
            base.groupby(
                'stay_id',
                sort=False
            )
        ):
            rewards = (
                group[
                    'reward'
                ]
                .to_numpy()
            )

            discounts = (
                gamma
                ** np.arange(
                    len(rewards)
                )
            )

            observed_returns.append(
                np.sum(
                    discounts
                    * rewards
                )
            )

        clinician_value = (
            float(
                np.mean(
                    observed_returns
                )
            )
        )

        for (
            policy_name,
            target_action_prob
        ) in target_probs.items():

            temp = base.copy()

            temp[
                'target_prob'
            ] = np.clip(
                target_action_prob,
                min_prob,
                1.0
            )

            log_weights = []
            returns = []
            stay_ids = []

            for stay_id, group in (
                temp.groupby(
                    'stay_id',
                    sort=False
                )
            ):
                log_ratio = (
                    np.log(
                        group[
                            'target_prob'
                        ].to_numpy()
                    )
                    -
                    np.log(
                        group[
                            'behavior_prob'
                        ].to_numpy()
                    )
                )

                log_weight = (
                    np.sum(
                        log_ratio
                    )
                )

                rewards = (
                    group[
                        'reward'
                    ]
                    .to_numpy()
                )

                discounts = (
                    gamma
                    ** np.arange(
                        len(rewards)
                    )
                )

                G = np.sum(
                    discounts
                    * rewards
                )

                log_weights.append(
                    log_weight
                )

                returns.append(
                    G
                )

                stay_ids.append(
                    stay_id
                )

            log_weights = np.asarray(
                log_weights
            )

            returns = np.asarray(
                returns
            )

            shifted = (
                log_weights
                -
                np.max(
                    log_weights
                )
            )

            weights = np.exp(
                shifted
            )

            weight_sum = (
                weights.sum()
            )

            if weight_sum == 0:
                normalized_weights = (
                    np.ones_like(
                        weights
                    )
                    / len(weights)
                )
            else:
                normalized_weights = (
                    weights
                    / weight_sum
                )

            wis = float(
                np.sum(
                    normalized_weights
                    * returns
                )
            )

            ess = float(
                1.0
                /
                np.sum(
                    normalized_weights
                    ** 2
                )
            )

            summary_rows.append(
                {
                    'Policy':
                        policy_name,

                    'WIS':
                        wis,

                    'ESS':
                        ess,

                    'Trajectories':
                        len(
                            returns
                        ),

                    'Clinician Observed Return':
                        clinician_value,

                    'Temperature':
                        temperature,

                    'Max Normalized Weight':
                        float(
                            normalized_weights
                            .max()
                        )
                }
            )

            for (
                stay_id,
                log_weight,
                norm_weight,
                G
            ) in zip(
                stay_ids,
                log_weights,
                normalized_weights,
                returns
            ):
                trajectory_rows.append(
                    {
                        'Policy':
                            policy_name,

                        'stay_id':
                            stay_id,

                        'Log Weight':
                            log_weight,

                        'Normalized Weight':
                            norm_weight,

                        'Return':
                            G
                    }
                )

        summary = pd.DataFrame(
            summary_rows
        )

        trajectories = pd.DataFrame(
            trajectory_rows
        )

        summary.to_csv(
            self.eval_dir
            / "wis_summary.csv",
            index=False
        )

        trajectories.to_csv(
            self.eval_dir
            / "wis_trajectory_weights.csv",
            index=False
        )

        labels = [
            'Clinician',
            'HRL',
            'Only Q',
            'Only Prospect'
        ]

        values = [
            clinician_value,
            float(
                summary.loc[
                    summary[
                        'Policy'
                    ]
                    == 'HRL',
                    'WIS'
                ].iloc[0]
            ),
            float(
                summary.loc[
                    summary[
                        'Policy'
                    ]
                    == 'Only Q',
                    'WIS'
                ].iloc[0]
            ),
            float(
                summary.loc[
                    summary[
                        'Policy'
                    ]
                    == 'Only Prospect',
                    'WIS'
                ].iloc[0]
            )
        ]

        self._save_bar(
            labels,
            values,
            'Weighted Importance Sampling',
            'Estimated Discounted Return',
            self.eval_dir
            / "wis_policy_value.png"
        )

        return summary

    def mortality_vs_q_value(
        self,
        policy_df,
        n_bins=10
    ):
        patient_df = (
            policy_df
            .groupby(
                'stay_id',
                as_index=False
            )
            .agg(
                high_q_value=(
                    'high_q_value',
                    'mean'
                ),
                mortality=(
                    'mortality',
                    'max'
                )
            )
        )

        try:
            patient_df[
                'Q Bin'
            ] = pd.qcut(
                patient_df[
                    'high_q_value'
                ],
                q=n_bins,
                duplicates='drop'
            )
        except ValueError:
            patient_df[
                'Q Bin'
            ] = pd.cut(
                patient_df[
                    'high_q_value'
                ],
                bins=n_bins
            )

        result = (
            patient_df
            .groupby(
                'Q Bin',
                observed=True
            )
            .agg(
                Mean_Q=(
                    'high_q_value',
                    'mean'
                ),
                Mortality_Rate=(
                    'mortality',
                    'mean'
                ),
                Patients=(
                    'stay_id',
                    'nunique'
                )
            )
            .reset_index(
                drop=True
            )
        )

        result.to_csv(
            self.eval_dir
            / "mortality_vs_q_value.csv",
            index=False
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.plot(
            result[
                'Mean_Q'
            ],
            result[
                'Mortality_Rate'
            ],
            marker='o'
        )

        ax.set_xlabel(
            'Mean High-Level Q Value'
        )

        ax.set_ylabel(
            'Mortality Rate'
        )

        ax.set_title(
            'Mortality vs High-Level Q Value'
        )

        fig.tight_layout()

        fig.savefig(
            self.eval_dir
            / "mortality_vs_q_value.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        return result

    def save_policy_value(
        self,
        wis_summary
    ):
        clinician_value = float(
            wis_summary[
                'Clinician Observed Return'
            ]
            .iloc[0]
        )

        rows = [
            {
                'Policy':
                    'Clinician',
                'Policy Value':
                    clinician_value,
                'Method':
                    'Observed discounted return'
            }
        ]

        for _, row in (
            wis_summary.iterrows()
        ):
            rows.append(
                {
                    'Policy':
                        row[
                            'Policy'
                        ],
                    'Policy Value':
                        row[
                            'WIS'
                        ],
                    'Method':
                        'WIS'
                }
            )

        result = pd.DataFrame(
            rows
        )

        result.to_csv(
            self.eval_dir
            / "policy_value.csv",
            index=False
        )

        self._save_bar(
            result[
                'Policy'
            ].tolist(),
            result[
                'Policy Value'
            ].tolist(),
            'Policy Value',
            'Discounted Return',
            self.eval_dir
            / "policy_value.png"
        )

        return result

    def _action_mortality_table(
        self,
        policy_df,
        action_col,
        prefix
    ):
        temp = (
            policy_df[
                [
                    'stay_id',
                    action_col,
                    'mortality'
                ]
            ]
            .drop_duplicates(
                subset=[
                    'stay_id',
                    action_col
                ]
            )
            .copy()
        )

        temp[
            'IV Level'
        ] = (
            temp[
                action_col
            ]
            // 5
        )

        temp[
            'Vasopressor Level'
        ] = (
            temp[
                action_col
            ]
            % 5
        )

        result = (
            temp
            .groupby(
                [
                    'IV Level',
                    'Vasopressor Level'
                ],
                as_index=False
            )
            .agg(
                Mortality_Rate=(
                    'mortality',
                    'mean'
                ),
                Patients=(
                    'stay_id',
                    'nunique'
                )
            )
        )

        result.to_csv(
            self.eval_dir
            / f"{prefix}_action_mortality.csv",
            index=False
        )

        matrix = (
            result
            .pivot(
                index='IV Level',
                columns=
                    'Vasopressor Level',
                values=
                    'Mortality_Rate'
            )
            .reindex(
                index=
                    range(5),
                columns=
                    range(5)
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        im = ax.imshow(
            matrix.to_numpy(),
            aspect='auto',
            vmin=0,
            vmax=1
        )

        ax.set_xticks(
            range(5)
        )

        ax.set_yticks(
            range(5)
        )

        ax.set_xlabel(
            'Vasopressor Level'
        )

        ax.set_ylabel(
            'IV Fluid Level'
        )

        ax.set_title(
            f'{prefix} Action Level vs Mortality'
        )

        for i in range(5):
            for j in range(5):

                value = (
                    matrix
                    .iloc[i, j]
                )

                if pd.notna(
                    value
                ):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha='center',
                        va='center'
                    )

        fig.colorbar(
            im,
            ax=ax,
            label='Mortality Rate'
        )

        fig.tight_layout()

        fig.savefig(
            self.eval_dir
            / f"{prefix}_action_mortality.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        return result

    def action_level_vs_mortality(
        self,
        policy_df
    ):
        clinician = (
            self
            ._action_mortality_table(
                policy_df,
                'clinician_action',
                'Clinician'
            )
        )

        hrl = (
            self
            ._action_mortality_table(
                policy_df,
                'hrl_action',
                'HRL'
            )
        )

        return {
            'Clinician':
                clinician,
            'HRL':
                hrl
        }

    def compare_only_q_prospect(
        self,
        policy_df,
        wis_summary
    ):
        clinician = (
            policy_df[
                'clinician_action'
            ]
            .to_numpy()
        )

        nonzero_mask = (
            clinician != 0
        )

        rows = []

        for (
            name,
            action_col
        ) in [
            (
                'Only Q',
                'q_action'
            ),
            (
                'Only Prospect',
                'prospect_action'
            )
        ]:

            pred = (
                policy_df[
                    action_col
                ]
                .to_numpy()
            )

            overall = (
                pred == clinician
            ).mean()

            if (
                nonzero_mask.sum()
                > 0
            ):
                nonzero = (
                    pred[
                        nonzero_mask
                    ]
                    ==
                    clinician[
                        nonzero_mask
                    ]
                ).mean()
            else:
                nonzero = np.nan

            wis_row = (
                wis_summary[
                    wis_summary[
                        'Policy'
                    ]
                    == name
                ]
            )

            wis = (
                wis_row[
                    'WIS'
                ].iloc[0]
                if len(
                    wis_row
                ) > 0
                else np.nan
            )

            ess = (
                wis_row[
                    'ESS'
                ].iloc[0]
                if len(
                    wis_row
                ) > 0
                else np.nan
            )

            rows.append(
                {
                    'Policy':
                        name,

                    'Agreement':
                        overall,

                    'Nonzero Agreement':
                        nonzero,

                    'Action0 Ratio':
                        (
                            pred == 0
                        ).mean(),

                    'WIS':
                        wis,

                    'ESS':
                        ess
                }
            )

        result = pd.DataFrame(
            rows
        )

        result.to_csv(
            self.eval_dir
            / "only_q_vs_only_prospect.csv",
            index=False
        )

        x = np.arange(
            len(result)
        )

        width = 0.35

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.bar(
            x - width / 2,
            result[
                'Agreement'
            ],
            width,
            label='Overall'
        )

        ax.bar(
            x + width / 2,
            result[
                'Nonzero Agreement'
            ],
            width,
            label='Nonzero'
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            result[
                'Policy'
            ]
        )

        ax.set_ylabel(
            'Agreement'
        )

        ax.set_title(
            'Only Q vs Only Prospect'
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            self.eval_dir
            / "only_q_vs_only_prospect.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        return result

    def record_hyperparameter_result(
        self,
        params,
        test_high,
        test_low,
        wis_summary
    ):
        path = (
            self.hyper_dir
            / "hyperparameter_results.csv"
        )

        row = dict(
            params
        )

        row[
            'hierarchy_agreement'
        ] = test_high.get(
            'hierarchy_agreement',
            np.nan
        )

        row[
            'hierarchy_nonzero_agreement'
        ] = test_high.get(
            'hierarchy_nonzero_agreement',
            np.nan
        )

        row[
            'option_agreement'
        ] = test_high.get(
            'option_agreement',
            np.nan
        )

        row[
            'q_agreement'
        ] = test_low.get(
            'q_agreement',
            np.nan
        )

        row[
            'prospect_agreement'
        ] = test_low.get(
            'p_agreement',
            np.nan
        )

        row[
            'q_nonzero_agreement'
        ] = test_low.get(
            'q_nonzero_agreement',
            np.nan
        )

        row[
            'prospect_nonzero_agreement'
        ] = test_low.get(
            'p_nonzero_agreement',
            np.nan
        )

        for policy_name in [
            'HRL',
            'Only Q',
            'Only Prospect'
        ]:

            subset = (
                wis_summary[
                    wis_summary[
                        'Policy'
                    ]
                    == policy_name
                ]
            )

            if len(
                subset
            ) > 0:

                key = (
                    policy_name
                    .replace(
                        ' ',
                        '_'
                    )
                    .lower()
                )

                row[
                    f"{key}_wis"
                ] = (
                    subset[
                        'WIS'
                    ]
                    .iloc[0]
                )

                row[
                    f"{key}_ess"
                ] = (
                    subset[
                        'ESS'
                    ]
                    .iloc[0]
                )

        row[
            'timestamp'
        ] = pd.Timestamp.now(
        ).isoformat()

        new_df = pd.DataFrame(
            [row]
        )

        if path.exists():

            old_df = pd.read_csv(
                path
            )

            result = pd.concat(
                [
                    old_df,
                    new_df
                ],
                ignore_index=True
            )

        else:
            result = new_df

        result.to_csv(
            path,
            index=False
        )

        self.plot_hyperparameter_results(
            result
        )

        return result

    def plot_hyperparameter_results(
        self,
        df
    ):
        parameter_columns = [
            'gamma',
            'lambda_pt',
            'cql_weight',
            'high_cql_weight',
            'low_level_epochs',
            'high_level_epochs'
        ]

        metric_columns = [
            'hrl_wis',
            'hierarchy_agreement',
            'hierarchy_nonzero_agreement',
            'option_agreement'
        ]

        for param in (
            parameter_columns
        ):

            if (
                param
                not in df.columns
            ):
                continue

            if (
                df[
                    param
                ].nunique()
                < 2
            ):
                continue

            for metric in (
                metric_columns
            ):

                if (
                    metric
                    not in df.columns
                ):
                    continue

                grouped = (
                    df
                    .groupby(
                        param,
                        as_index=False
                    )[
                        metric
                    ]
                    .mean()
                    .sort_values(
                        param
                    )
                )

                fig, ax = plt.subplots(
                    figsize=(8, 6)
                )

                ax.plot(
                    grouped[
                        param
                    ],
                    grouped[
                        metric
                    ],
                    marker='o'
                )

                ax.set_xlabel(
                    param
                )

                ax.set_ylabel(
                    metric
                )

                ax.set_title(
                    f'{metric} vs {param}'
                )

                fig.tight_layout()

                fig.savefig(
                    self.hyper_dir
                    / (
                        f"{metric}"
                        f"_vs_"
                        f"{param}.png"
                    ),
                    dpi=300,
                    bbox_inches='tight'
                )

                plt.close(fig)

    def visualize_latent_tsne_full(
        self,
        sepsis_encoder,
        data_loader
    ):
        sepsis_encoder.encoder.eval()

        mu_list = []
        labels_list = []

        with torch.no_grad():

            for (
                batch_X,
                _,
                batch_s
            ) in data_loader:

                batch_X = batch_X.to(
                    sepsis_encoder.device
                )

                mu, _ = (
                    sepsis_encoder
                    .encoder(
                        batch_X
                    )
                )

                mu_list.append(
                    mu.cpu().numpy()
                )

                labels_list.append(
                    batch_s.numpy()
                )

        mu_all = np.concatenate(
            mu_list,
            axis=0
        )

        labels_all = np.concatenate(
            labels_list,
            axis=0
        )

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30.0,
            max_iter=1000
        )

        tsne_results = (
            tsne.fit_transform(
                mu_all
            )
        )

        tsne_df = pd.DataFrame(
            {
                'TSNE1':
                    tsne_results[
                        :, 0
                    ],

                'TSNE2':
                    tsne_results[
                        :, 1
                    ],

                'Stage':
                    labels_all
            }
        )

        tsne_df.to_csv(
            self.sde_dir
            / "latent_tsne.csv",
            index=False
        )

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        stages = np.unique(
            labels_all
        )

        for stage in stages:

            mask = (
                labels_all
                == stage
            )

            ax.scatter(
                tsne_results[
                    mask,
                    0
                ],
                tsne_results[
                    mask,
                    1
                ],
                alpha=0.7,
                s=30,
                label=
                    f'Stage {stage}'
            )

        ax.set_title(
            't-SNE Visualization of Sepsis States'
        )

        ax.set_xlabel(
            't-SNE Dimension 1'
        )

        ax.set_ylabel(
            't-SNE Dimension 2'
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            self.sde_dir
            / "latent_tsne.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        return tsne_df

    def get_hmm_state_stats(
        self,
        df_hmm
    ):
        results = []

        for state_id, group in (
            df_hmm.groupby(
                'hmm_state'
            )
        ):

            res = {
                'HMM State':
                    f"State {state_id}"
            }

            n_obs = len(group)

            res[
                'Total Observations'
            ] = (
                f"{n_obs} "
                f"({n_obs / len(df_hmm) * 100:.1f}%)"
            )

            res[
                'Unique Patients'
            ] = str(
                group[
                    'stay_id'
                ].nunique()
            )

            def get_iqr(
                series
            ):
                s = pd.to_numeric(
                    series,
                    errors='coerce'
                ).dropna()

                if len(s) == 0:
                    return "N/A"

                return (
                    f"{s.median():.1f} "
                    f"("
                    f"{s.quantile(0.25):.1f}"
                    f"-"
                    f"{s.quantile(0.75):.1f}"
                    f")"
                )

            res[
                'SOFA Score, Median(IQR)'
            ] = get_iqr(
                group.get(
                    'sofa_score',
                    pd.Series(
                        dtype=float
                    )
                )
            )

            res[
                'PaO2, Median(IQR)'
            ] = get_iqr(
                group[
                    'pao2'
                ]
            )

            res[
                'FiO2, Median(IQR)'
            ] = get_iqr(
                group[
                    'fio2'
                ]
            )

            res[
                'Platelets, Median(IQR)'
            ] = get_iqr(
                group[
                    'platelets'
                ]
            )

            res[
                'Bilirubin, Median(IQR)'
            ] = get_iqr(
                group[
                    'bilirubin'
                ]
            )

            res[
                'Creatinine, Median(IQR)'
            ] = get_iqr(
                group[
                    'creatinine'
                ]
            )

            res[
                'Lactate, Median(IQR)'
            ] = get_iqr(
                group.get(
                    'lactate',
                    pd.Series(
                        dtype=float
                    )
                )
            )

            res[
                'GCS, Median(IQR)'
            ] = get_iqr(
                group[
                    'gcs'
                ]
            )

            results.append(
                res
            )

        return (
            pd.DataFrame(
                results
            )
            .set_index(
                'HMM State'
            )
            .T
        )

    def save_hmm_results(
        self,
        df_hmm,
        hmm_module=None
    ):
        stats = (
            self
            .get_hmm_state_stats(
                df_hmm
            )
        )

        stats.to_csv(
            self.hmm_dir
            / "hmm_state_stats.csv"
        )

        state_counts = (
            df_hmm[
                'hmm_state'
            ]
            .value_counts()
            .sort_index()
        )

        state_counts.to_csv(
            self.hmm_dir
            / "hmm_state_counts.csv"
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.bar(
            state_counts
            .index
            .astype(str),
            state_counts
            .values
        )

        ax.set_xlabel(
            'HMM State'
        )

        ax.set_ylabel(
            'Observations'
        )

        ax.set_title(
            'HMM State Distribution'
        )

        fig.tight_layout()

        fig.savefig(
            self.hmm_dir
            / "hmm_state_distribution.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        if hmm_module is not None:

            model = None

            for attr in [
                'model',
                'hmm',
                'hmm_model'
            ]:

                if hasattr(
                    hmm_module,
                    attr
                ):
                    candidate = getattr(
                        hmm_module,
                        attr
                    )

                    if hasattr(
                        candidate,
                        'transmat_'
                    ):
                        model = candidate
                        break

            if model is not None:

                trans = np.asarray(
                    model.transmat_
                )

                trans_df = pd.DataFrame(
                    trans,
                    index=[
                        f"State {i}"
                        for i
                        in range(
                            trans.shape[0]
                        )
                    ],
                    columns=[
                        f"State {i}"
                        for i
                        in range(
                            trans.shape[1]
                        )
                    ]
                )

                trans_df.to_csv(
                    self.hmm_dir
                    / "hmm_transition_matrix.csv"
                )

                fig, ax = plt.subplots(
                    figsize=(7, 6)
                )

                im = ax.imshow(
                    trans,
                    vmin=0,
                    vmax=1
                )

                ax.set_xticks(
                    range(
                        trans.shape[1]
                    )
                )

                ax.set_yticks(
                    range(
                        trans.shape[0]
                    )
                )

                ax.set_xlabel(
                    'Next State'
                )

                ax.set_ylabel(
                    'Current State'
                )

                ax.set_title(
                    'HMM Transition Matrix'
                )

                for i in range(
                    trans.shape[0]
                ):
                    for j in range(
                        trans.shape[1]
                    ):
                        ax.text(
                            j,
                            i,
                            f"{trans[i, j]:.2f}",
                            ha='center',
                            va='center'
                        )

                fig.colorbar(
                    im,
                    ax=ax
                )

                fig.tight_layout()

                fig.savefig(
                    self.hmm_dir
                    / "hmm_transition_matrix.png",
                    dpi=300,
                    bbox_inches='tight'
                )

                plt.close(fig)

        return stats

    def extract_and_visualize_fg_norms_by_stage(
        self,
        encoder_module,
        data_loader
    ):
        encoder_module.encoder.eval()
        encoder_module.sde.eval()

        f_l2_list = []
        f_max_list = []
        g_l2_list = []
        g_max_list = []
        stage_list = []

        device = (
            encoder_module.device
        )

        with torch.no_grad():

            for (
                batch_X,
                _,
                batch_s
            ) in data_loader:

                batch_X = (
                    batch_X.to(
                        device
                    )
                )

                mu, _ = (
                    encoder_module
                    .encoder(
                        batch_X
                    )
                )

                t_zero = torch.zeros(
                    mu.shape[0],
                    1,
                    device=device
                )

                ty = torch.cat(
                    [
                        t_zero,
                        mu
                    ],
                    dim=-1
                )

                f_val = (
                    encoder_module
                    .sde
                    .f_net(ty)
                )

                g_val = (
                    encoder_module
                    .sde
                    .g_net(ty)
                )

                f_l2_list.extend(
                    torch.norm(
                        f_val,
                        p=2,
                        dim=-1
                    )
                    .cpu()
                    .numpy()
                )

                f_max_list.extend(
                    torch.norm(
                        f_val,
                        p=float(
                            'inf'
                        ),
                        dim=-1
                    )
                    .cpu()
                    .numpy()
                )

                g_l2_list.extend(
                    torch.norm(
                        g_val,
                        p=2,
                        dim=-1
                    )
                    .cpu()
                    .numpy()
                )

                g_max_list.extend(
                    torch.norm(
                        g_val,
                        p=float(
                            'inf'
                        ),
                        dim=-1
                    )
                    .cpu()
                    .numpy()
                )

                stage_list.extend(
                    batch_s
                    .cpu()
                    .numpy()
                )

        df_norms = pd.DataFrame(
            {
                'Stage':
                    stage_list,

                'F_L2_Norm':
                    f_l2_list,

                'F_Max_Norm':
                    f_max_list,

                'G_L2_Norm':
                    g_l2_list,

                'G_Max_Norm':
                    g_max_list
            }
        )

        df_norms.to_csv(
            self.sde_dir
            / "fg_norms_by_stage.csv",
            index=False
        )

        stats = (
            df_norms
            .groupby(
                'Stage'
            )
            .describe()
            .T
        )

        stats.to_csv(
            self.sde_dir
            / "fg_norms_descriptive_stats.csv"
        )

        metrics = [
            (
                'F_L2_Norm',
                'Drift f - L2 Norm'
            ),
            (
                'F_Max_Norm',
                'Drift f - Max Norm'
            ),
            (
                'G_L2_Norm',
                'Diffusion g - L2 Norm'
            ),
            (
                'G_Max_Norm',
                'Diffusion g - Max Norm'
            )
        ]

        stages = sorted(
            df_norms[
                'Stage'
            ].unique()
        )

        for col, title in metrics:

            data = [
                df_norms.loc[
                    df_norms[
                        'Stage'
                    ]
                    == stage,
                    col
                ].values
                for stage
                in stages
            ]

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            ax.boxplot(
                data,
                tick_labels=[
                    str(stage)
                    for stage
                    in stages
                ]
            )

            ax.set_xlabel(
                'Stage'
            )

            ax.set_ylabel(
                col
            )

            ax.set_title(
                title
            )

            fig.tight_layout()

            fig.savefig(
                self.sde_dir
                / f"{col}.png",
                dpi=300,
                bbox_inches='tight'
            )

            plt.close(fig)

        return df_norms

    def visualize_sde_trajectory_pca(
        self,
        encoder_module,
        data_loader,
        num_samples=5,
        time_steps=50
    ):
        encoder_module.encoder.eval()
        encoder_module.sde.eval()

        device = (
            encoder_module.device
        )

        (
            batch_X,
            _,
            batch_s
        ) = next(
            iter(
                data_loader
            )
        )

        num_samples = min(
            num_samples,
            batch_X.shape[0]
        )

        batch_X = (
            batch_X[
                :num_samples
            ]
            .to(device)
        )

        stages = (
            batch_s[
                :num_samples
            ]
            .numpy()
        )

        with torch.no_grad():

            mu, _ = (
                encoder_module
                .encoder(
                    batch_X
                )
            )

            ts = torch.linspace(
                0.0,
                1.0,
                steps=time_steps,
                device=device
            )

            y0_aug = torch.cat(
                [
                    mu,
                    torch.zeros(
                        mu.shape[0],
                        1,
                        device=device
                    )
                ],
                dim=-1
            )

            y_aug_ts = (
                torchsde.sdeint(
                    encoder_module.sde,
                    y0_aug,
                    ts,
                    method='euler',
                    dt=0.05
                )
            )

            z_traj = (
                y_aug_ts[
                    :, :, :-1
                ]
                .cpu()
                .numpy()
            )

        z_traj_flat = (
            z_traj.reshape(
                -1,
                z_traj.shape[-1]
            )
        )

        pca = PCA(
            n_components=2
        )

        z_traj_2d = (
            pca
            .fit_transform(
                z_traj_flat
            )
            .reshape(
                time_steps,
                num_samples,
                2
            )
        )

        trajectory_rows = []

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        for i in range(
            num_samples
        ):

            traj = (
                z_traj_2d[
                    :, i, :
                ]
            )

            ax.plot(
                traj[
                    :, 0
                ],
                traj[
                    :, 1
                ],
                linewidth=2,
                label=(
                    f'Patient {i + 1} '
                    f'(Stage {stages[i]})'
                )
            )

            ax.scatter(
                traj[
                    0, 0
                ],
                traj[
                    0, 1
                ],
                marker='o',
                s=100
            )

            ax.scatter(
                traj[
                    -1, 0
                ],
                traj[
                    -1, 1
                ],
                marker='X',
                s=120
            )

            for t in range(
                time_steps
            ):
                trajectory_rows.append(
                    {
                        'Patient':
                            i + 1,

                        'Stage':
                            stages[i],

                        'Time Step':
                            t,

                        'PC1':
                            traj[
                                t,
                                0
                            ],

                        'PC2':
                            traj[
                                t,
                                1
                            ]
                    }
                )

        ax.set_title(
            '2D PCA Projection of SDE Trajectories'
        )

        ax.set_xlabel(
            (
                f"PC1 "
                f"("
                f"{pca.explained_variance_ratio_[0]:.1%}"
                f")"
            )
        )

        ax.set_ylabel(
            (
                f"PC2 "
                f"("
                f"{pca.explained_variance_ratio_[1]:.1%}"
                f")"
            )
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            self.sde_dir
            / "sde_trajectory_pca.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close(fig)

        trajectory_df = pd.DataFrame(
            trajectory_rows
        )

        trajectory_df.to_csv(
            self.sde_dir
            / "sde_trajectory_pca.csv",
            index=False
        )

        return trajectory_df