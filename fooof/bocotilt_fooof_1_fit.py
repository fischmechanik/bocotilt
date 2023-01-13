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

# Define paths
path_in = "/mnt/data_dump/bocotilt/8_ged77/component_time_series/"
path_out = "/mnt/data_dump/bocotilt/8_fooof/fooof_models/"
path_fooof = "/home/plkn/Downloads/fooof/"

# Append fooof to sys path
sys.path.append(path_fooof)

# Import fooof
import fooof

# Set sampling rate
srate = 200

# List of datasets
datasets = glob.glob(f"{path_in}/*.mat")

# Loop datasets
for counter_subject, dataset in enumerate(datasets):

    # Talk
    print(f"subject {counter_subject + 1}/{len(datasets)}")

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load time series
    eeg_data = np.squeeze(scipy.io.loadmat(dataset)["cmp_time_series"])

    # Load times
    eeg_times = np.squeeze(scipy.io.loadmat(dataset)["times"])

    # Load trialinfo
    df_trialinfo = pd.DataFrame(scipy.io.loadmat(dataset)["trialinfo"])

    # Set column labels
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

    # Get dims
    n_times, n_epochs = eeg_data.shape

    # Get time window idx
    idx_timewins = (
        (eeg_times >= -800) & (eeg_times < 0),
        (eeg_times >= 0) & (eeg_times < 800),
        (eeg_times >= 800) & (eeg_times < 1600),
    )

    # Init output
    output = {
        "baseline": [],
        "ct_interval": [],
        "post_target": [],
        "trialinfo": df_trialinfo,
    }

    # Loop timewins
    for timewin_nr, timewin_idx in enumerate(idx_timewins):

        # Select data and transpose (time needs to be last dimension...)
        tmp = eeg_data[timewin_idx, :].T

        # Compute spectrum
        spectra, fooof_freqs = mne.time_frequency.psd_array_welch(
            tmp,
            srate,
            fmin=1,
            fmax=40,
            n_fft=1024,
            n_per_seg=1024,
            n_jobs=-2,
            average="mean",
            window="hamming",
        )

        # Loop trials
        for epoch in range(n_epochs):

            # Initialize FOOOF
            fm = fooof.FOOOF()

            # Select spectrum
            fooof_spectrum = spectra[epoch, :]

            # Set the frequency range to fit the fooof model
            fooof_freq_range = [1, 40]

            # Report: fit the model
            fm.fit(fooof_freqs, fooof_spectrum, fooof_freq_range)

            output[list(output.keys())[timewin_nr]].append(fm)

    # Specify out file name
    out_file = os.path.join(path_out, f"{id_string}_fooof_models.joblib")

    # Save
    joblib.dump(output, out_file)
