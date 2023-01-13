#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import sys
import statsmodels.stats.anova

# Define paths
path_in = "/mnt/data_dump/bocotilt/8_fooof/fooof_models/"
path_fooof = "/home/plkn/Downloads/fooof/"

# Append fooof to sys path
sys.path.append(path_fooof)

# Import fooof
import fooof

# List of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Init pandas stuff
cols = [
    "id",
    "condition",
    "reward",
    "switch",
    "ap_off_bl",
    "ap_off_ct",
    "ap_off_pt",
    "ap_exp_bl",
    "ap_exp_ct",
    "ap_exp_pt",
    "theta_cf_bl",
    "theta_cf_ct",
    "theta_cf_pt",
    "theta_pw_bl",
    "theta_pw_ct",
    "theta_pw_pt",
    "theta_bw_bl",
    "theta_bw_ct",
    "theta_bw_pt",
    "alpha_cf_bl",
    "alpha_cf_ct",
    "alpha_cf_pt",
    "alpha_pw_bl",
    "alpha_pw_ct",
    "alpha_pw_pt",
    "alpha_bw_bl",
    "alpha_bw_ct",
    "alpha_bw_pt",
]
df = pd.DataFrame(columns=cols, index=range(len(datasets) * 4))
df_idx_counter = -1

# Loop datasets
for counter_subject, dataset in enumerate(datasets):

    # Load dataset
    fooof_data = joblib.load(dataset)

    # Get condition idx
    tinf = fooof_data["trialinfo"]

    condition_idx = [
        tinf.index[(tinf["bonus"] == 0) & (tinf["task_switch"] == 0)].tolist(),
        tinf.index[(tinf["bonus"] == 0) & (tinf["task_switch"] == 1)].tolist(),
        tinf.index[(tinf["bonus"] == 1) & (tinf["task_switch"] == 0)].tolist(),
        tinf.index[(tinf["bonus"] == 1) & (tinf["task_switch"] == 1)].tolist(),
    ]

    # Condition labels
    condition_labels = ["std_rep", "std_swi", "bon_rep", "bon_swi"]

    # Loop conditions
    for condition_nr, cidx in enumerate(condition_idx):

        # Populate df
        df_idx_counter += 1
        df.loc[df_idx_counter]["id"] = int(dataset.split("/")[-1][:2])
        df.loc[df_idx_counter]["condition"] = condition_labels[condition_nr]
        df.loc[df_idx_counter]["reward"] = condition_labels[condition_nr][0:3]
        df.loc[df_idx_counter]["switch"] = condition_labels[condition_nr][4:7]

        # Iterate time windows
        for tw_idx, tw in enumerate(["baseline", "ct_interval", "post_target"]):

            # Get a more compact time window identifier
            tw_compact = ["bl", "ct", "pt"][tw_idx]

            # Parameter matrix
            fooof_params = []

            # Loop trials of condition
            for idx in cidx:

                # Get fitted fooof model
                fm = fooof_data[tw][idx]

                # Get aperiodic params
                ap_off, ap_exp = fm.aperiodic_params_

                # Get theta periodics
                theta_cf, theta_pw, theta_bw = fooof.analysis.get_band_peak_fm(
                    fm, [4, 8]
                )

                # Get alpha periodics
                alpha_cf, alpha_pw, alpha_bw = fooof.analysis.get_band_peak_fm(
                    fm, [8, 13]
                )

                # Append
                fooof_params.append(
                    [
                        ap_off,
                        ap_exp,
                        theta_cf,
                        theta_pw,
                        theta_bw,
                        alpha_cf,
                        alpha_pw,
                        alpha_bw,
                    ]
                )

            # Stack and average
            (
                ap_off,
                ap_exp,
                theta_cf,
                theta_pw,
                theta_bw,
                alpha_cf,
                alpha_pw,
                alpha_bw,
            ) = np.nanmean(np.stack(fooof_params), axis=0)

            # More pupulating going on...
            df.loc[df_idx_counter][f"ap_off_{tw_compact}"] = ap_off
            df.loc[df_idx_counter][f"ap_exp_{tw_compact}"] = ap_exp
            df.loc[df_idx_counter][f"theta_cf_{tw_compact}"] = theta_cf
            df.loc[df_idx_counter][f"theta_pw_{tw_compact}"] = theta_pw
            df.loc[df_idx_counter][f"theta_bw_{tw_compact}"] = theta_bw
            df.loc[df_idx_counter][f"alpha_cf_{tw_compact}"] = alpha_cf
            df.loc[df_idx_counter][f"alpha_pw_{tw_compact}"] = alpha_pw
            df.loc[df_idx_counter][f"alpha_bw_{tw_compact}"] = alpha_bw

# Add event related parameters (baseline substracted)
df["er_ap_off_ct"] = df["ap_off_ct"] - df["ap_off_bl"]
df["er_ap_off_pt"] = df["ap_off_pt"] - df["ap_off_bl"]

df["er_ap_exp_ct"] = df["ap_exp_ct"] - df["ap_exp_bl"]
df["er_ap_exp_pt"] = df["ap_exp_pt"] - df["ap_exp_bl"]

df["er_theta_cf_ct"] = df["theta_cf_ct"] - df["theta_cf_bl"]
df["er_theta_cf_pt"] = df["theta_cf_pt"] - df["theta_cf_bl"]

df["er_theta_pw_ct"] = df["theta_pw_ct"] - df["theta_pw_bl"]
df["er_theta_pw_pt"] = df["theta_pw_pt"] - df["theta_pw_bl"]

df["er_theta_bw_ct"] = df["theta_bw_ct"] - df["theta_bw_bl"]
df["er_theta_bw_pt"] = df["theta_bw_pt"] - df["theta_bw_bl"]

df["er_alpha_cf_ct"] = df["alpha_cf_ct"] - df["alpha_cf_bl"]
df["er_alpha_cf_pt"] = df["alpha_cf_pt"] - df["alpha_cf_bl"]

df["er_alpha_pw_ct"] = df["alpha_pw_ct"] - df["alpha_pw_bl"]
df["er_alpha_pw_pt"] = df["alpha_pw_pt"] - df["alpha_pw_bl"]

df["er_alpha_bw_ct"] = df["alpha_bw_ct"] - df["alpha_bw_bl"]
df["er_alpha_bw_pt"] = df["alpha_bw_pt"] - df["alpha_bw_bl"]

measure = "er_theta_cf_pt"

# Plot RTs (still including incorrect?)
g = sns.catplot(
    x="reward",
    y=measure,
    hue="switch",
    capsize=0.05,
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
g.despine(left=True)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df,
    depvar=measure,
    subject="id",
    within=["reward", "switch"],
).fit()
print(anova_out)
