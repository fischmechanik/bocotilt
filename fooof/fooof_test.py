#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import pandas as pd
import joblib
import os
import sys
import scipy.io
import matplotlib.pyplot as plt
import statsmodels.stats.anova
import seaborn as sns

# Define paths
path_clean_data = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_fooof = "/home/plkn/fooof_main/"

# Append fooof to sys path
sys.path.append(path_fooof)

# Import fooof
import fooof

# List of datasets
datasets = glob.glob(f"{path_clean_data}/*_erp.set")

# List of ids to exclude
ids_to_exclude = []

# Init pandas stuff
cols = [
    "id",
    "condition",
    "reward",
    "switch",
    "fm_index_bl",
    "fm_index_ct",
    "fm_index_pt",
    "n_theta_peaks_bl",
    "n_theta_peaks_ct",
    "n_theta_peaks_pt",
    "theta_cf_bl",
    "theta_cf_ct",
    "theta_cf_pt",
]
df = pd.DataFrame(columns=cols, index=range((len(datasets) - len(ids_to_exclude)) * 4))
df_idx_counter = -1
fm_counter = -1

# List for fooof models
fooof_models = []

# Loop datasets
for counter_subject, dataset in enumerate(datasets):

    # Talk
    print(f"subject {counter_subject + 1}/{len(datasets)}")

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Exclude subjects
    if id_string in ids_to_exclude:
        continue

    # Load data
    eeg = scipy.io.loadmat(dataset)
               
    # Unpack           
    eeg_data, eeg_times, srate = eeg["data"], np.squeeze(eeg["times"]), np.squeeze(eeg["srate"])                       
    
    # Get FCz data (trials x times)
    fcz_data = eeg_data[126, :, :].T
    
    # Create trialinfo as dataframe
    df_trialinfo = pd.DataFrame(eeg["trialinfo"])

    # Set trialinfo column labels
    df_trialinfo.columns = [
        "id",
        "block",
        "trial_nr",
        "bonus",
        "tilt_task",
        "cue_ax",
        "target_red_left",
        "distractor_red_left",
        "response_interference",
        "task_switch",
        "prev_switch",
        "prev_accuracy",
        "correct_response",
        "response_side",
        "rt",
        "rt_thresh_color",
        "rt_thresh_tilt",
        "accuracy",
        "position_color",
        "position_tilt",
        "position_target",
        "position_distractor",
        "sequence_position",
    ]

    # Get idx of conditions
    condition_idx = [
        df_trialinfo.index[
            (df_trialinfo["bonus"] == 0) & (df_trialinfo["task_switch"] == 0)
        ].tolist(),
        df_trialinfo.index[
            (df_trialinfo["bonus"] == 0) & (df_trialinfo["task_switch"] == 1)
        ].tolist(),
        df_trialinfo.index[
            (df_trialinfo["bonus"] == 1) & (df_trialinfo["task_switch"] == 0)
        ].tolist(),
        df_trialinfo.index[
            (df_trialinfo["bonus"] == 1) & (df_trialinfo["task_switch"] == 1)
        ].tolist(),
    ]

    # Condition labels
    condition_labels = ["std_rep", "std_swi", "bon_rep", "bon_swi"]

    # Loop conditions
    for condition_nr, cidx in enumerate(condition_idx):

        # Populate df
        df_idx_counter += 1
        df.loc[df_idx_counter]["id"] = int(dataset.split("/")[-1][2:4])
        df.loc[df_idx_counter]["condition"] = condition_labels[condition_nr]
        df.loc[df_idx_counter]["reward"] = condition_labels[condition_nr][0:3]
        df.loc[df_idx_counter]["switch"] = condition_labels[condition_nr][4:7]

        # Get time window idx
        idx_timewins = (
            (eeg_times >= -700) & (eeg_times < -100),
            (eeg_times >= 100) & (eeg_times < 700),
            (eeg_times >= 900) & (eeg_times < 1500),
        )

        # Loop timewins
        for tw_idx, tw in enumerate(["baseline", "ct_interval", "post_target"]):

            # Get a more compact time window identifier
            tw_compact = ["bl", "ct", "pt"][tw_idx]

            # Select trials
            tmp_fcz = fcz_data[cidx, :].copy()

            # Select timewin
            tmp_fcz = tmp_fcz[:, np.squeeze(idx_timewins[tw_idx])].copy()

            # Compute spectrumfcz
            spectrum_fcz, fooof_freqs = mne.time_frequency.psd_array_welch(
                tmp_fcz,
                srate,
                fmin=0.01,
                fmax=40,
                n_fft=1024,
                n_per_seg=120,
                n_overlap=80,
                n_jobs=-2,
                average="mean",
                window="hamming",
            )

            # Average spectra
            spectrum_fcz = spectrum_fcz.mean(axis=0)

            # Initialize FOOOF
            fm = fooof.FOOOF(
                peak_threshold=0.1, peak_width_limits=[1, 4], max_n_peaks=100
            )

            # Set the frequency range to fit the fooof model
            fooof_freq_range = [1, 30]

            # Report: fit the model
            fm_counter += 1
            fm.fit(np.squeeze(fooof_freqs), spectrum_fcz, fooof_freq_range)
            df.loc[df_idx_counter][f"fm_index_{tw_compact}"] = fm_counter
            
            # Save fooof
            fooof_models.append(fm)

            # Collect theta peaks
            theta_peaks = []
            for peak_params in fm.peak_params_:
                if (peak_params[0] >= 4) & (peak_params[0] <= 9):
                    theta_peaks.append(peak_params)

            # Save observed number of peaks
            df.loc[df_idx_counter][f"n_theta_peaks_{tw_compact}"] = len(theta_peaks)

            # If more than 0 theta peaks
            if len(theta_peaks) > 0:

                # Get index of max amplitude theta peak
                maxidx = np.argmax(np.array(theta_peaks)[:, 1])

                # Save peak frequency
                df.loc[df_idx_counter][f"theta_cf_{tw_compact}"] = theta_peaks[maxidx][
                    0
                ]

# Baseline some measures
df["er_theta_cf_ct"] = df["theta_cf_ct"] - df["theta_cf_bl"]
df["er_theta_cf_pt"] = df["theta_cf_pt"] - df["theta_cf_bl"]

# Select measure
measure = "theta_cf_pt"

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


# Inspect
fooof_idx = 5
fooof_models[fooof_idx].plot(plot_peaks="shade", peak_kwargs={"color": "green"})
fooof_models[fooof_idx].peak_params_
