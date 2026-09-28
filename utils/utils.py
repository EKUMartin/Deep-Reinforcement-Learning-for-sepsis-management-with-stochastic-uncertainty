from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_clinical_reward_columns(
    df,
    terminal_reward_value=5.0,
):
    """

    project convention:
    survival == 0 : survivor
    survival != 0 : death

    reward_step = clip(SOFA_t - SOFA_t+1, -4, 4) / 4
    terminal = +5 survivor / -5 death
    clinical_reward = reward_step + is_last * terminal
    """
    out = df.copy()

    sofa_delta = (
        out["sofa_score"].astype(float)
        - out["sofa_score_next"].astype(float)
    )
    out["reward_step"] = sofa_delta.clip(
        lower=-4.0, upper=4.0
    ) / 4.0

    out["terminal_outcome_reward"] = np.where(
        out["survival"].astype(int) == 0,
        float(terminal_reward_value),
        -float(terminal_reward_value),
    )

    out["clinical_reward"] = (
        out["reward_step"]
        + out["is_last"].astype(float)
        * out["terminal_outcome_reward"]
    )

    return out


class DiffusionUtils:
    """
    Single SDE encoder에서 latent z와 diffusion g를 추출한다.
    """

    def __init__(
        self,
        encoder_module,
        device,
        eps=1e-6,
    ):
        self.encoder_module = encoder_module
        self.device = device
        self.eps = eps
        self.g_mean = None
        self.g_std = None

    def get_latent(self, x):
        self.encoder_module.encoder.eval()
        with torch.no_grad():
            mu, _ = self.encoder_module.encoder(x)
        return mu

    def get_raw(self, z):
        self.encoder_module.sde.eval()
        with torch.no_grad():
            t_zero = torch.zeros_like(z[:, :1])
            ty = torch.cat([t_zero, z], dim=1)
            g_val = (
                self.encoder_module.sde.g_net(ty)
                + 1e-3
            )
            return g_val.max(dim=1).values

    def scale(self, g_raw):
        if self.g_mean is None or self.g_std is None:
            raise RuntimeError(
                "fit_stats()"
            )

        return torch.sigmoid(
            (g_raw - self.g_mean)
            / (self.g_std + self.eps)
        )

    def fit_stats(self, loader):
        values = []

        self.encoder_module.encoder.eval()
        self.encoder_module.sde.eval()

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                z = self.get_latent(x)
                g_raw = self.get_raw(z)
                values.append(g_raw.cpu())

        values = torch.cat(values, dim=0)

        self.g_mean = values.mean().item()
        self.g_std = max(
            values.std(unbiased=False).item(),
            1e-6,
        )

        return self.g_mean, self.g_std

    def set_stats(self, g_mean, g_std):
        self.g_mean = float(g_mean)
        self.g_std = max(float(g_std), 1e-6)


class DiffusionValidator:
    def __init__(
        self,
        project_root,
        encoder_module,
        scaler,
        device,
        dynamic_features,
        next_dynamic_features,
        g_mean,
        g_std,
        interval=4,
        analysis_window_hours=96,
        random_seed=42,
    ):
        self.project_root = Path(project_root)
        self.encoder_module = encoder_module
        self.scaler = scaler
        self.device = device
        self.dynamic_features = list(dynamic_features)
        self.next_dynamic_features = list(
            next_dynamic_features
        )
        self.g_mean = float(g_mean)
        self.g_std = max(float(g_std), 1e-6)
        self.interval = interval
        self.analysis_window_hours = (
            analysis_window_hours
        )
        self.random_seed = random_seed

        self.output_dir = (
            self.project_root
            / "results"
            / "diffusion_validation"
        )
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.encoder_module.encoder.eval()
        self.encoder_module.sde.eval()
        self.encoder_module.decoder.eval()

    @staticmethod
    def _safe_spearman(x, y):
        from scipy.stats import spearmanr

        temp = pd.DataFrame(
            {
                "x": np.asarray(x),
                "y": np.asarray(y),
            }
        )
        temp = (
            temp.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(temp) < 3:
            return np.nan, np.nan

        rho, p_value = spearmanr(
            temp["x"].to_numpy(),
            temp["y"].to_numpy(),
        )
        return float(rho), float(p_value)

    def _encode_mu(
        self,
        x_np,
        batch_size=4096,
    ):
        out = []

        with torch.no_grad():
            for start in range(
                0,
                len(x_np),
                batch_size,
            ):
                x = torch.as_tensor(
                    x_np[
                        start:
                        start + batch_size
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                mu, _ = (
                    self.encoder_module
                    .encoder(x)
                )
                out.append(
                    mu.cpu().numpy()
                )

        return np.concatenate(
            out,
            axis=0,
        )

    def _diffusion_from_z(
        self,
        z_np,
        batch_size=4096,
    ):
        raw = []
        scaled = []

        with torch.no_grad():
            for start in range(
                0,
                len(z_np),
                batch_size,
            ):
                z = torch.as_tensor(
                    z_np[
                        start:
                        start + batch_size
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                t_zero = torch.zeros_like(
                    z[:, :1]
                )

                g_val = (
                    self.encoder_module
                    .sde
                    .g_net(
                        torch.cat(
                            [t_zero, z],
                            dim=1,
                        )
                    )
                    + 1e-3
                )

                g_raw = g_val.max(
                    dim=1
                ).values

                g_scaled = torch.sigmoid(
                    (
                        g_raw
                        - self.g_mean
                    )
                    /
                    (
                        self.g_std
                        + 1e-6
                    )
                )

                raw.append(
                    g_raw.cpu().numpy()
                )
                scaled.append(
                    g_scaled.cpu().numpy()
                )

        return (
            np.concatenate(raw),
            np.concatenate(scaled),
        )

    def _mc_one_step_prediction(
        self,
        z_current,
        x_next,
        mc_samples=30,
        batch_size=256,
    ):
        pred_mean_list = []
        error_list = []
        dispersion_list = []

        with torch.no_grad():
            for start in range(
                0,
                len(z_current),
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    len(z_current),
                )

                z = torch.as_tensor(
                    z_current[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                next_x = torch.as_tensor(
                    x_next[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                b = z.shape[0]
                z_repeat = (
                    z[:, None, :]
                    .expand(
                        b,
                        mc_samples,
                        z.shape[1],
                    )
                    .reshape(
                        b * mc_samples,
                        z.shape[1],
                    )
                    .contiguous()
                )

                z_traj, _ = (
                    self.encoder_module
                    .sde(
                        z_repeat,
                        self.encoder_module.ts,
                    )
                )

                z_pred = z_traj[-1]

                x_pred = (
                    self.encoder_module
                    .decoder(z_pred)
                    .reshape(
                        b,
                        mc_samples,
                        len(
                            self.dynamic_features
                        ),
                    )
                )

                x_pred_mean = x_pred.mean(
                    dim=1
                )

                pred_error = torch.sqrt(
                    torch.mean(
                        (
                            next_x
                            - x_pred_mean
                        ) ** 2,
                        dim=1,
                    )
                )

                pred_std = x_pred.std(
                    dim=1,
                    unbiased=False,
                )

                pred_dispersion = torch.sqrt(
                    torch.mean(
                        pred_std ** 2,
                        dim=1,
                    )
                )

                pred_mean_list.append(
                    x_pred_mean
                    .cpu()
                    .numpy()
                )
                error_list.append(
                    pred_error
                    .cpu()
                    .numpy()
                )
                dispersion_list.append(
                    pred_dispersion
                    .cpu()
                    .numpy()
                )

        return (
            np.concatenate(
                pred_mean_list,
                axis=0,
            ),
            np.concatenate(
                error_list,
            ),
            np.concatenate(
                dispersion_list,
            ),
        )

    def _mahalanobis_ood(
        self,
        z_train,
        z_test,
    ):
        latent_mean = z_train.mean(
            axis=0
        )

        covariance = np.cov(
            z_train,
            rowvar=False,
        )

        covariance = (
            covariance
            + np.eye(
                covariance.shape[0]
            )
            * 1e-4
        )

        inv_covariance = np.linalg.pinv(
            covariance
        )

        centered = (
            z_test
            - latent_mean
        )

        mahalanobis_sq = np.einsum(
            "bi,ij,bj->b",
            centered,
            inv_covariance,
            centered,
        )

        return np.sqrt(
            np.maximum(
                mahalanobis_sq,
                0,
            )
        )

    def _save_plots(
        self,
        raw_df,
        perturbation_df,
    ):
        import matplotlib.pyplot as plt

        plot_specs = [
            (
                "g_scaled",
                "prediction_error",
                "Diffusion uncertainty",
                "4h prediction RMSE",
                "diffusion_vs_prediction_error.png",
            ),
            (
                "g_scaled",
                "predictive_dispersion",
                "Diffusion uncertainty",
                "MC predictive dispersion",
                "diffusion_vs_predictive_dispersion.png",
            ),
            (
                "g_scaled",
                "ood_distance",
                "Diffusion uncertainty",
                "Latent Mahalanobis OOD",
                "diffusion_vs_ood.png",
            ),
        ]

        for (
            x_col,
            y_col,
            x_label,
            y_label,
            filename,
        ) in plot_specs:

            temp = (
                raw_df[
                    [x_col, y_col]
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .copy()
            )

            if len(temp) == 0:
                continue

            try:
                temp["decile"] = (
                    pd.qcut(
                        temp[x_col],
                        q=10,
                        labels=False,
                        duplicates="drop",
                    )
                    + 1
                )

                decile = (
                    temp.groupby(
                        "decile",
                        as_index=False,
                    )
                    .agg(
                        mean_x=(
                            x_col,
                            "mean",
                        ),
                        mean_y=(
                            y_col,
                            "mean",
                        ),
                    )
                )

                fig, ax = plt.subplots(
                    figsize=(8, 6)
                )
                ax.plot(
                    decile["mean_x"],
                    decile["mean_y"],
                    marker="o",
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(
                    self.output_dir
                    / filename,
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

            except Exception:
                pass

        if (
            perturbation_df is not None
            and len(perturbation_df) > 0
        ):
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )
            ax.plot(
                perturbation_df[
                    "noise_level"
                ],
                perturbation_df[
                    "mean_g_scaled"
                ],
                marker="o",
            )
            ax.set_xlabel(
                "Gaussian noise level "
                "(scaled feature space)"
            )
            ax.set_ylabel(
                "Mean diffusion uncertainty"
            )
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                self.output_dir
                / "synthetic_noise_vs_diffusion.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    def run_all(
        self,
        df_train,
        df_test,
        mc_samples=30,
        max_validation_transitions=15000,
        max_ood_train_samples=50000,
    ):
        np.random.seed(
            self.random_seed
        )
        torch.manual_seed(
            self.random_seed
        )

        df_transition = (
            df_test
            .sort_values(
                ["stay_id", "charttime"]
            )
            .reset_index(drop=True)
            .copy()
        )

        if "hours_from_onset" in (
            df_transition.columns
        ):
            df_transition = (
                df_transition[
                    df_transition[
                        "hours_from_onset"
                    ].between(
                        0,
                        self.analysis_window_hours,
                        inclusive="both",
                    )
                ]
                .copy()
            )

        if "transition_hours" in (
            df_transition.columns
        ):
            df_transition = (
                df_transition[
                    np.isclose(
                        df_transition[
                            "transition_hours"
                        ],
                        self.interval,
                    )
                ]
                .copy()
            )

        required = (
            self.dynamic_features
            + self.next_dynamic_features
            + [
                "sofa_score",
                "stay_id",
            ]
        )

        df_transition = (
            df_transition
            .dropna(
                subset=required
            )
            .reset_index(drop=True)
        )

        if (
            len(df_transition)
            > max_validation_transitions
        ):
            df_transition = (
                df_transition
                .sample(
                    max_validation_transitions,
                    random_state=
                        self.random_seed,
                )
                .sort_values(
                    ["stay_id", "charttime"]
                )
                .reset_index(drop=True)
            )

        print(
            "\n"
            "============================================================"
        )
        print(
            "DIFFUSION VALIDATION"
        )
        print(
            "============================================================"
        )
        print(
            f"Window: onset 0-"
            f"{self.analysis_window_hours}h"
        )
        print(
            "Transitions:",
            len(df_transition),
        )
        print(
            "Stays:",
            df_transition[
                "stay_id"
            ].nunique(),
        )
        print(
            "MC samples:",
            mc_samples,
        )

        x_current = (
            self.scaler
            .transform(
                df_transition[
                    self.dynamic_features
                ].values
            )
        )

        x_next = (
            self.scaler
            .transform(
                df_transition[
                    self.next_dynamic_features
                ].values
            )
        )

        z_current = self._encode_mu(
            x_current
        )

        g_raw, g_scaled = (
            self._diffusion_from_z(
                z_current
            )
        )

        (
            x_pred_mean,
            prediction_error,
            predictive_dispersion,
        ) = self._mc_one_step_prediction(
            z_current=z_current,
            x_next=x_next,
            mc_samples=mc_samples,
        )

        per_feature_rmse = np.sqrt(
            np.mean(
                (
                    x_next
                    - x_pred_mean
                ) ** 2,
                axis=0,
            )
        )

        pd.DataFrame(
            {
                "feature":
                    self.dynamic_features,
                "standardized_rmse":
                    per_feature_rmse,
            }
        ).to_csv(
            self.output_dir
            / "per_feature_prediction_rmse.csv",
            index=False,
        )

        df_ood_train = df_train.copy()

        if (
            len(df_ood_train)
            > max_ood_train_samples
        ):
            df_ood_train = (
                df_ood_train
                .sample(
                    max_ood_train_samples,
                    random_state=
                        self.random_seed,
                )
                .copy()
            )

        x_train_ood = (
            self.scaler
            .transform(
                df_ood_train[
                    self.dynamic_features
                ].values
            )
        )

        z_train_ood = self._encode_mu(
            x_train_ood
        )

        ood_distance = (
            self._mahalanobis_ood(
                z_train=z_train_ood,
                z_test=z_current,
            )
        )

        raw_df = pd.DataFrame(
            {
                "stay_id":
                    df_transition[
                        "stay_id"
                    ].to_numpy(),
                "charttime":
                    df_transition[
                        "charttime"
                    ].to_numpy(),
                "sofa_score":
                    df_transition[
                        "sofa_score"
                    ].to_numpy(),
                "g_raw":
                    g_raw,
                "g_scaled":
                    g_scaled,
                "prediction_error":
                    prediction_error,
                "predictive_dispersion":
                    predictive_dispersion,
                "ood_distance":
                    ood_distance,
            }
        )

        if "mortality" in (
            df_transition.columns
        ):
            raw_df["mortality"] = (
                df_transition[
                    "mortality"
                ].to_numpy()
            )

        raw_df.to_csv(
            self.output_dir
            / "diffusion_validation_raw.csv",
            index=False,
        )

        raw_df["g_decile"] = (
            pd.qcut(
                raw_df["g_scaled"],
                q=10,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        decile_df = (
            raw_df
            .groupby(
                "g_decile",
                as_index=False,
            )
            .agg(
                mean_g=(
                    "g_scaled",
                    "mean",
                ),
                mean_prediction_error=(
                    "prediction_error",
                    "mean",
                ),
                mean_predictive_dispersion=(
                    "predictive_dispersion",
                    "mean",
                ),
                mean_ood_distance=(
                    "ood_distance",
                    "mean",
                ),
                mean_sofa=(
                    "sofa_score",
                    "mean",
                ),
                n=(
                    "stay_id",
                    "size",
                ),
                unique_stays=(
                    "stay_id",
                    "nunique",
                ),
            )
        )

        decile_df.to_csv(
            self.output_dir
            / "diffusion_deciles.csv",
            index=False,
        )

        (
            rho_g_error,
            p_g_error,
        ) = self._safe_spearman(
            raw_df["g_scaled"],
            raw_df["prediction_error"],
        )

        (
            rho_g_dispersion,
            p_g_dispersion,
        ) = self._safe_spearman(
            raw_df["g_scaled"],
            raw_df["predictive_dispersion"],
        )

        (
            rho_dispersion_error,
            p_dispersion_error,
        ) = self._safe_spearman(
            raw_df["predictive_dispersion"],
            raw_df["prediction_error"],
        )

        (
            rho_g_ood,
            p_g_ood,
        ) = self._safe_spearman(
            raw_df["g_scaled"],
            raw_df["ood_distance"],
        )

        (
            rho_sofa_g,
            p_sofa_g,
        ) = self._safe_spearman(
            raw_df["sofa_score"],
            raw_df["g_scaled"],
        )

        (
            rho_sofa_error,
            p_sofa_error,
        ) = self._safe_spearman(
            raw_df["sofa_score"],
            raw_df["prediction_error"],
        )

        adjusted_beta = np.nan
        adjusted_p = np.nan
        adjusted_ci_low = np.nan
        adjusted_ci_high = np.nan

        try:
            import statsmodels.api as sm
            from sklearn.preprocessing import StandardScaler

            reg_df = (
                raw_df[
                    [
                        "stay_id",
                        "prediction_error",
                        "g_scaled",
                        "sofa_score",
                        "ood_distance",
                    ]
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .copy()
            )

            scaler_reg = StandardScaler()

            scaled = (
                scaler_reg
                .fit_transform(
                    reg_df[
                        [
                            "prediction_error",
                            "g_scaled",
                            "sofa_score",
                            "ood_distance",
                        ]
                    ]
                )
            )

            reg_df[
                "error_z"
            ] = scaled[:, 0]

            reg_df[
                "g_z"
            ] = scaled[:, 1]

            reg_df[
                "sofa_z"
            ] = scaled[:, 2]

            reg_df[
                "ood_z"
            ] = scaled[:, 3]

            x_reg = sm.add_constant(
                reg_df[
                    [
                        "g_z",
                        "sofa_z",
                        "ood_z",
                    ]
                ]
            )

            model = (
                sm.OLS(
                    reg_df["error_z"],
                    x_reg,
                )
                .fit(
                    cov_type="cluster",
                    cov_kwds={
                        "groups":
                            reg_df[
                                "stay_id"
                            ]
                    },
                )
            )

            adjusted_beta = float(
                model.params["g_z"]
            )
            adjusted_p = float(
                model.pvalues["g_z"]
            )

            ci = (
                model
                .conf_int()
                .loc["g_z"]
            )

            adjusted_ci_low = float(
                ci.iloc[0]
            )
            adjusted_ci_high = float(
                ci.iloc[1]
            )

            with open(
                self.output_dir
                / "adjusted_regression.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    model.summary().as_text()
                )

        except Exception as e:
            with open(
                self.output_dir
                / "adjusted_regression_error.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(e))

        rng = np.random.default_rng(
            self.random_seed
        )

        perturb_n = min(
            10000,
            len(df_transition),
        )

        perturb_idx = rng.choice(
            len(df_transition),
            size=perturb_n,
            replace=False,
        )

        x_base = x_current[
            perturb_idx
        ]

        perturb_rows = []

        for noise_level in [
            0.0,
            0.05,
            0.10,
            0.20,
            0.50,
            1.00,
        ]:
            noise_rng = (
                np.random.default_rng(
                    self.random_seed
                )
            )

            x_noise = (
                x_base
                + noise_rng.normal(
                    0.0,
                    noise_level,
                    size=x_base.shape,
                )
            )

            z_noise = self._encode_mu(
                x_noise
            )

            _, g_noise = (
                self._diffusion_from_z(
                    z_noise
                )
            )

            perturb_rows.append(
                {
                    "noise_level":
                        noise_level,
                    "mean_g_scaled":
                        float(
                            np.mean(g_noise)
                        ),
                    "median_g_scaled":
                        float(
                            np.median(g_noise)
                        ),
                    "std_g_scaled":
                        float(
                            np.std(g_noise)
                        ),
                }
            )

        perturbation_df = pd.DataFrame(
            perturb_rows
        )

        perturbation_df.to_csv(
            self.output_dir
            / "synthetic_perturbation.csv",
            index=False,
        )

        self._save_plots(
            raw_df=raw_df,
            perturbation_df=
                perturbation_df,
        )

        summary = pd.DataFrame(
            [
                {
                    "n_transitions":
                        len(raw_df),
                    "n_stays":
                        raw_df[
                            "stay_id"
                        ].nunique(),
                    "mc_samples":
                        mc_samples,

                    "spearman_g_prediction_error":
                        rho_g_error,
                    "p_g_prediction_error":
                        p_g_error,

                    "spearman_g_predictive_dispersion":
                        rho_g_dispersion,
                    "p_g_predictive_dispersion":
                        p_g_dispersion,

                    "spearman_dispersion_prediction_error":
                        rho_dispersion_error,
                    "p_dispersion_prediction_error":
                        p_dispersion_error,

                    "spearman_g_ood":
                        rho_g_ood,
                    "p_g_ood":
                        p_g_ood,

                    "spearman_sofa_g":
                        rho_sofa_g,
                    "p_sofa_g":
                        p_sofa_g,

                    "spearman_sofa_prediction_error":
                        rho_sofa_error,
                    "p_sofa_prediction_error":
                        p_sofa_error,

                    "adjusted_g_beta":
                        adjusted_beta,
                    "adjusted_g_p":
                        adjusted_p,
                    "adjusted_g_ci_low":
                        adjusted_ci_low,
                    "adjusted_g_ci_high":
                        adjusted_ci_high,
                }
            ]
        )

        summary.to_csv(
            self.output_dir
            / "DIFFUSION_VALIDATION_SUMMARY.csv",
            index=False,
        )

        print(
            "\nDiffusion vs prediction error:",
            rho_g_error,
        )
        print(
            "Diffusion vs predictive dispersion:",
            rho_g_dispersion,
        )
        print(
            "Predictive dispersion vs prediction error:",
            rho_dispersion_error,
        )
        print(
            "Diffusion vs OOD:",
            rho_g_ood,
        )
        print(
            "SOFA vs diffusion:",
            rho_sofa_g,
        )
        print(
            "SOFA vs prediction error:",
            rho_sofa_error,
        )
        print(
            "Adjusted diffusion beta:",
            adjusted_beta,
        )

        return summary
