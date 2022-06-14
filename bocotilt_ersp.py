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
path_out = "/mnt/data_dump/bocotilt/4_ersp/"

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
    eeg_epochs = mne.io.read_epochs_eeglab(dataset)

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

    # Remove practice trials and incorrect trials and first of sequence
    idx_to_drop = (
        (df_trialinfo["block"] <= 4)
        | (df_trialinfo["log_accuracy"] != 1)
        | (df_trialinfo["sequence_position"] == 1)
    ).to_numpy()
    df_trialinfo = df_trialinfo.loc[np.invert(idx_to_drop), :]
    eeg_epochs.drop(idx_to_drop)

    # Get condition indices
    condition_idx = {
        "standard_repeat": (df_trialinfo["bonus"] == 0)
        & (df_trialinfo["task_switch"] == 0),
        "standard_switch": (df_trialinfo["bonus"] == 0)
        & (df_trialinfo["task_switch"] == 1),
        "bonus_repeat": (df_trialinfo["bonus"] == 1)
        & (df_trialinfo["task_switch"] == 0),
        "bonus_switch": (df_trialinfo["bonus"] == 1)
        & (df_trialinfo["task_switch"] == 1),
    }

    # Initialize result dict
    tf_data = {
        "conditions": [],
        "power": [],
        "itc": [],
    }

    # Loop factor level combinations
    for cond in condition_idx:

        # Select epochs
        eeg_epochs_cond = eeg_epochs[condition_idx[cond]]

        # Perform time-frequency analysis and apply baseline
        n_freqs = 50
        tf_freqs = np.linspace(2, 20, n_freqs)
        tf_cycles = np.linspace(3, 10, n_freqs)
        power, itc = mne.time_frequency.tfr_morlet(
            eeg_epochs_cond, tf_freqs, n_cycles=tf_cycles, n_jobs=-2, decim=2,
        )
        power.apply_baseline(mode="logratio", baseline=(-0.5, -0.2))

        # Crop
        power = power.crop(tmin=-0.5, tmax=2)
        itc = itc.crop(tmin=-0.5, tmax=2)

        # Collect across condition combinations
        tf_data["conditions"].append(cond)
        tf_data["power"].append(power)
        tf_data["itc"].append(itc)

    # Collect across subjects
    tf_datasets.append(tf_data)

# Save
out_file = os.path.join(path_out, f"tf_datasets_task_switch_bonus.joblib")
joblib.dump(tf_datasets, out_file)

