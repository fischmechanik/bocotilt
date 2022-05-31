#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import joblib
import imblearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Path variables
path_in = "/mnt/data2/bocotilt/2_autocleaned/"
path_out = "/mnt/data2/bocotilt/rsa_matrices/"

# Function for bootstrapping averages
def do_some_binning(data_in, n_batch, n_averages):
    averages = np.zeros((n_averages, data_in.shape[1]))
    for n in range(n_averages):
        averages[n, :] = data_in[np.random.choice(data_in.shape[0], n_batch), :].mean(
            axis=0
        )
    return averages


# Prediction function
def predict_stuff(clf, X, y):
    acc = np.zeros((len(X),))
    for idx in range(n_times):
        acc[idx] = sklearn.metrics.accuracy_score(y[idx], clf.predict(X[idx]),)
    return acc


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Specify out file name
    out_file = os.path.join(path_out, f"{id_string}_rsa_matrices.joblib")

    # Skip if file exists already
    out_file = os.path.join(path_out, f"{id_string}_rsa_matrices.joblib")
    if os.path.isfile(out_file):
        continue

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Perform single trial time-frequency analysis
    tf_freqs = np.linspace(2, 30, 20)
    tf_cycles = np.linspace(3, 12, 20)
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

    # Clean up
    del eeg_epochs

    # Prune in time
    pruneframes = 40
    tf_times = tf_epochs.times[pruneframes:-pruneframes]
    tf_data = tf_epochs.data[:, :, :, pruneframes:-pruneframes]

    # Clean up
    del tf_epochs

    # Load trialinfo
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )

    # Exclude trials: Practice-block trials and incorrect trials
    idx_to_keep = (trialinfo[:, 1] >= 5) & (trialinfo[:, 16] == 1)
    trialinfo = trialinfo[idx_to_keep, :]
    tf_data = tf_data[idx_to_keep, :, :, :]

    # get dims
    n_trials, n_channels, n_freqs, n_times = tf_data.shape

    # Trialinfo cols:
    #  0: id
    #  1: block_nr
    #  2: trial_nr
    #  3: bonustrial  x
    #  4: tilt_task  x
    #  5: cue_ax  x
    #  6: target_red_left
    #  7: distractor_red_left
    #  8: response_interference
    #  9: task_switch
    # 10: correct_response
    # 11: response_side
    # 12: rt
    # 13: accuracy
    # 14: log_response_side x
    # 15: log_rt
    # 16: log_accuracy
    # 17: position_color
    # 18: position_tilt
    # 19: position_target   x
    # 20: position_distractor   x
    # 21: sequence_position
    # 22: sequence_length

    # Get trial indices
    idx_standard_color = (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 0)
    idx_standard_tilt = (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 1)
    idx_bonus_color = (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 0)
    idx_bonus_tilt = (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 1)

    # Iterate time
    standard_trained_clfs = []
    standard_test_data_X = []
    standard_test_data_y = []
    bonus_trained_clfs = []
    bonus_test_data_X = []
    bonus_test_data_y = []
    for time_idx, timeval in enumerate(tf_times):

        print(f"Training classifier {time_idx}/{tf_times.shape[0]}")

        # Data as trials x channels x frequencies
        timepoint_data = tf_data[:, :, :, time_idx]

        # Trials in rows
        timepoint_data_2d = timepoint_data.reshape((n_trials, n_channels * n_freqs))

        # Bin data
        n_bins = 300
        data_standard_color = do_some_binning(
            timepoint_data_2d[idx_standard_color, :], 20, n_bins
        )
        data_standard_tilt = do_some_binning(
            timepoint_data_2d[idx_standard_tilt, :], 20, n_bins
        )
        data_bonus_color = do_some_binning(
            timepoint_data_2d[idx_bonus_color, :], 20, n_bins
        )
        data_bonus_tilt = do_some_binning(
            timepoint_data_2d[idx_bonus_tilt, :], 20, n_bins
        )

        # Stack bins
        X_standard, X_bonus = (
            np.vstack((data_standard_color, data_standard_tilt)),
            np.vstack((data_bonus_color, data_bonus_tilt)),
        )

        # Get labels
        y_standard, y_bonus = (
            np.repeat([0, 1], n_bins),
            np.repeat([0, 1], n_bins),
        )

        # Split
        (
            X_standard_train,
            X_standard_test,
            y_standard_train,
            y_standard_test,
        ) = sklearn.model_selection.train_test_split(
            X_standard, y_standard, test_size=0.33, random_state=42
        )
        (
            X_bonus_train,
            X_bonus_test,
            y_bonus_train,
            y_bonus_test,
        ) = sklearn.model_selection.train_test_split(
            X_bonus, y_bonus, test_size=0.33, random_state=42
        )

        # Shuffle data
        X_standard_train, y_standard_train = sklearn.utils.shuffle(
            X_standard_train, y_standard_train
        )
        X_standard_test, y_standard_test = sklearn.utils.shuffle(
            X_standard_test, y_standard_test
        )
        X_bonus_train, y_bonus_train = sklearn.utils.shuffle(
            X_bonus_train, y_bonus_train
        )
        X_bonus_test, y_bonus_test = sklearn.utils.shuffle(X_bonus_test, y_bonus_test)

        # Keep test data
        standard_test_data_X.append(X_standard_test)
        standard_test_data_y.append(y_standard_test)
        bonus_test_data_X.append(X_bonus_test)
        bonus_test_data_y.append(y_bonus_test)

        # Fit models
        clf.fit(X_standard_train, y_standard_train)
        standard_trained_clfs.append(clf)
        
        
        clf.fit(X_bonus_train, y_bonus_train)
        bonus_trained_clfs.append(clf)

    # Clean up
    del tf_data

    # Init RSA matrices
    rsa_matrix_standard = np.zeros((n_times, n_times))
    rsa_matrix_bonus = np.zeros((n_times, n_times))

    # Make predictions
    acc_standard = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(predict_stuff)(clf, standard_test_data_X, standard_test_data_y)
        for clf in standard_trained_clfs
    )
    
    a = np.stack(acc_standard)

    aa = bb

