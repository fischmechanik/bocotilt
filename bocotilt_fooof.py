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
import matplotlib.pyplot as plt

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_fooof = "/home/plkn/Downloads/fooof/"

# Append fooof to sys path
sys.path.append(path_fooof)

# Import fooof
import fooof

# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")

# A result list
results = []

# Loop datasets
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset)

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

    # Remove practice trials and incorrect trials and first of sequence
    idx_to_drop = (
        (df_trialinfo["block"] <= 4)
        | (df_trialinfo["accuracy"] != 1)
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
    result_data = {
        "a": [],
        "b": [],
        "c": [],
    }

    pf = []
    
    # Loop factor level combinations
    for cond in condition_idx:

        # Select epochs
        eeg_epochs_cond = eeg_epochs[condition_idx[cond]]

        # Get dims
        n_epochs, n_channel, n_times = eeg_epochs_cond.get_data().shape

        # Select data
        idx_time = (eeg_epochs_cond.times > 0) & (eeg_epochs_cond.times < 0.8)
        tmp = eeg_epochs_cond.get_data()[:, :, idx_time]

        # Compute spectrum
        spectra, fooof_freqs = mne.time_frequency.psd_array_welch(
            tmp,
            100,
            fmin=1,
            fmax=40,
            n_fft=1024,
            n_overlap=128,
            n_jobs=-2,
            average="mean",
            window="hamming",
        )

        # Initialize FOOOF
        fm = fooof.FOOOF()
        
        # Set the frequency range to fit the fooof model
        fooof_freq_range = [2, 40]

        theta_peaks = []

        # Loop trials
        for epoch in range(n_epochs):
            
            # Select spectrum 
            fooof_spectrum = spectra[epoch, 126, :]

            # Report: fit the model, print the resulting parameters, and plot the reconstruction
            fm.fit(fooof_freqs, fooof_spectrum, fooof_freq_range)
            
            fm.plot()
        
            # get theta peaks
            theta_peaks.append(fooof.analysis.get_band_peak_fm(fm, [2, 7]))
        
        # Collect for each condition
        pf.append(np.nanmean(np.stack(theta_peaks)[:, 0]))
        
    aa=bb
        







