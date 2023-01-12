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

# Condition labels
condition_labels = ["std_rep", "std_swi", "bon_rep", "bon_swi"]

# Timewin labels
timewin_labels = ["baseline", "ct_interval", "post_target"]

# Init pandas stuff
cols = [
    "id",
    "condition",
    "reward",
    "switch",
    "timewin",
    "ap_off",
    "ap_exp",
    "theta_cf",
    "theta_pw",
    "theta_bw",
    "alpha_cf",
    "alpha_pw",
    "alpha_bw",
]
df = pd.DataFrame(columns=cols, index=range(len(datasets) * 4 * 3))
df_counter = -1

# Loop datasets
for counter_subject, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("/")[-1][:2]

    # Load dataset
    fooof_data = joblib.load(dataset)

    # Loop conditions
    for condition_label in condition_labels:

        # Loop conditions
        for timewin_label in timewin_labels:

            # Get single trial fooof data
            data_trials = fooof_data[f"fooof_{condition_label}_{timewin_label}"]

            # Parameter matrix
            fooof_params = []

            # Loop trials
            for fm in data_trials:

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

            # Populate df
            df_counter += 1
            df.loc[df_counter] = [
                int(id_string),
                condition_label,
                condition_label[0:3],
                condition_label[4:7],
                timewin_label,
                ap_off,
                ap_exp,
                theta_cf,
                theta_pw,
                theta_bw,
                alpha_cf,
                alpha_pw,
                alpha_bw,
            ]


# Plot RTs (still including incorrect?)
g = sns.catplot(
    x="timewin",
    y="theta_cf",
    hue="condition",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
g.despine(left=True)

anova_out_rt = statsmodels.stats.anova.AnovaRM(
    data=df,
    depvar="theta_cf",
    subject="id",
    within=["timewin", "reward", "switch"],
).fit()
print(anova_out_rt)
