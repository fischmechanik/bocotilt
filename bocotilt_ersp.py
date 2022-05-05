#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import pandas as pd
import itertools
import joblib
import os

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/6_tf_analysis/"

# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")

# Create a montage
standard_montage = mne.channels.make_standard_montage("standard_1005")

# Loop datasets
tf_datasets = []
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Rename channels and seet montage
    eeg_epochs.rename_channels({"O9": "I1", "O10": "I2"})
    eeg_epochs.set_montage(standard_montage)

    # Load trialinfo
    df_trialinfo = pd.read_csv(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", header=None
    )
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
        "correct_response",
        "response_side",
        "rt",
        "accuracy",
        "log_response_side",
        "log_rt",
        "log_accuracy",
        "position_color",
        "position_tilt",
        "position_target",
        "position_distractor",
        "sequence_position",
        "sequence_length",
    ]

    # Remove practice trials and incorrect trials
    idx_to_drop = (
        (df_trialinfo["block"] <= 4) | (df_trialinfo["log_accuracy"] != 1)
    ).to_numpy()
    df_trialinfo = df_trialinfo.loc[np.invert(idx_to_drop), :]
    eeg_epochs.drop(idx_to_drop)

    # Define experimental factors
    factors = ["block", "bonus"]

    # Get all factor levels
    factor_levels = [list(df_trialinfo[f].unique()) for f in factors]

    # Get all factor level combinations
    factor_level_combinations = list(itertools.product(*factor_levels))

    # Initialize result dict
    tf_data = {
        "factors": factors,
        "levels": factor_level_combinations,
        "power": [],
        "itc": [],
    }

    # Loop factor level combinations
    for comb in factor_level_combinations:

        # Boolean mask
        boolean_mask = np.all(
            np.stack(
                [
                    (df_trialinfo[f] == comb[f_idx]).to_numpy()
                    for f_idx, f in enumerate(factors)
                ]
            ),
            axis=0,
        )

        # Select epochs
        eeg_epochs_comb = eeg_epochs[boolean_mask]

        # Perform time-frequency analysis and apply baseline
        tf_freqs = np.linspace(2, 16, 20)
        tf_cycles = np.linspace(3, 12, 20)
        power, itc = mne.time_frequency.tfr_morlet(
            eeg_epochs, tf_freqs, n_cycles=tf_cycles, n_jobs=-2, decim=2,
        )
        power.apply_baseline(mode="logratio", baseline=(-0.5, -0.2))

        # Collect across condition combinations
        tf_data["power"].append(power)
        tf_data["itc"].append(itc)

    # Collect across subjects
    tf_datasets.append(tf_data)

# Save
factors_as_string = ""
for f in factors:
    factors_as_string = factors_as_string + "_" + f

out_file = os.path.join(path_out, f"tf_datasets{factors_as_string}.joblib")
joblib.dump(tf_datasets, out_file)


