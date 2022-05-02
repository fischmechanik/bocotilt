#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import sklearn.preprocessing
import sklearn.linear_model
import joblib
import os

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/5_regression/"

# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")

# Init stuff
linreg = sklearn.linear_model.LinearRegression()
scaler = sklearn.preprocessing.StandardScaler()

# Loop datasets
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Load trialinfo
    #  0: id
    #  1: block_nr
    #  2: trial_nr
    #  3: bonustrial
    #  4: tilt_task
    #  5: cue_ax
    #  6: target_red_left
    #  7: distractor_red_left
    #  8: response_interference
    #  9: task_switch
    # 10: correct_response
    # 11: response_side
    # 12: rt
    # 13: accuracy
    # 14: log_response_side
    # 15: log_rt
    # 16: log_accuracy
    # 17: position_color
    # 18: position_tilt
    # 19: position_target
    # 20: position_distractor
    # 21: sequence_position
    # 22: sequence_length
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )

    # Create event codes
    combined_codes = np.array(
        [
            int(
                str(int(trialinfo[x, 3]))  # Bonustrial
                + str(int(trialinfo[x, 21]))  # sequence position
            )
            for x in range(trialinfo.shape[0])
        ]
    )

    # Exclude practice blocks
    eeg_epochs.drop(trialinfo[:, 1] <= 4)
    trialinfo = trialinfo[trialinfo[:, 1] > 4, :]

    # Perform single trial time-frequency analysis
    tf_freqs = np.linspace(2, 20, 15)
    tf_cycles = np.linspace(3, 12, 15)
    tf_epochs = mne.time_frequency.tfr_morlet(
        eeg_epochs,
        tf_freqs,
        n_cycles=tf_cycles,
        average=False,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )
    
    # Apply baseline procedure
    tf_epochs.apply_baseline(mode="logratio", baseline=(-0.5, -0.2))
    
    # Prune in time
    pruneframes = (100, 100)
    tf_times = tf_epochs.times[pruneframes[0] : -pruneframes[1]]
    tf_data = tf_epochs.data[:, :, :, pruneframes[0] : -pruneframes[1]]

    # Get predictors for tot analysis (tot experiment, tot sequence)
    X = scaler.fit_transform(trialinfo[:, (2, 21)])

    # Get dimensions
    n_trials, n_channels, n_freqs, n_times = tf_data.shape

    # Reshape data
    y = tf_data.reshape((n_trials, -1))

    # Split standard and bonus trials
    idx_standard = trialinfo[:, 3] == 0
    idx_bonus = trialinfo[:, 3] == 1
    X_standard = X[idx_standard, :]
    X_bonus = X[idx_bonus, :]
    y_standard = y[idx_standard, :]
    y_bonus = y[idx_bonus, :]

    # Fit regression models
    coef_standard = linreg.fit(X_standard, y_standard).coef_
    coef_bonus = linreg.fit(X_bonus, y_bonus).coef_
    
    # Unpack and compile results
    res = {
        "coef_standard_trialnum": coef_standard[:, 0].reshape((n_channels, n_freqs, n_times)),
        "coef_standard_seqpos": coef_standard[:, 1].reshape((n_channels, n_freqs, n_times)),
        "coef_bonus_trialnum": coef_bonus[:, 0].reshape((n_channels, n_freqs, n_times)),
        "coef_bonus_seqpos": coef_bonus[:, 1].reshape((n_channels, n_freqs, n_times)),
    }

    # Save
    out_file = os.path.join(path_out, f"{id_string}_regression_data.joblib")
    joblib.dump(res, out_file)
    
# Save tf params
out_file = os.path.join(path_out, "tf_times.joblib")
joblib.dump(tf_times, out_file)
out_file = os.path.join(path_out, "tf_freqs.joblib")
joblib.dump(tf_freqs, out_file)

