#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import os
import joblib
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import scipy.signal
import scipy.io
import mne
import time

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/1_eeg_icset_tf/"
path_out = "/mnt/data_dump/bocotilt/3_decoded/"

# Define pruneframes (number of frames pruned at each side of epoch)
pruneframes = 100

# A smoothening function
def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# Filter-Hilbert function
def filter_hilbert(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    y = scipy.signal.filtfilt(b, a, data)
    y = scipy.signal.hilbert(y)
    return y


# Function for bootstrapping averages
def bootstrap_averages(data_in, n_batch, n_averages, n_features):
    averages = np.zeros((n_averages, n_features))
    for n in range(n_averages):
        averages[n, :] = data_in[np.random.choice(data_in.shape[0], n_batch), :].mean(
            axis=0
        )
    return averages


# Function performing random forest classification
# Works only for 2 classes, y-data must be coded using 0 and 1
def random_forest_classification(X, y, n_batch, n_averages_train, n_averages_test):

    # Init splitter
    n_splits = 4
    kf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)

    # Init classifier
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        random_state=42,
    )

    # Arrays for results
    acc_true = 0
    fmp_true = np.zeros((X.shape[1],))
    acc_fake = 0
    fmp_fake = np.zeros((X.shape[1],))

    # Iterate splits
    for kidx, (idx_train, idx_test) in enumerate(kf.split(X, y)):

        # Split data
        X_train, X_test = X[idx_train, :], X[idx_test, :]
        y_train, y_test = y[idx_train], y[idx_test]

        # Build test and training erps using bootstrapping
        X_train_0 = bootstrap_averages(
            X_train[y_train == 0, :], n_batch, n_averages_train, X.shape[1]
        )
        X_train_1 = bootstrap_averages(
            X_train[y_train == 1, :], n_batch, n_averages_train, X.shape[1]
        )
        X_test_0 = bootstrap_averages(
            X_test[y_test == 0, :], n_batch, n_averages_test, X.shape[1]
        )
        X_test_1 = bootstrap_averages(
            X_test[y_test == 1, :], n_batch, n_averages_test, X.shape[1]
        )

        # Stack averages
        X_train, X_test = (
            np.vstack((X_train_0, X_train_1)),
            np.vstack((X_test_0, X_test_1)),
        )

        # Training and test labels
        y_train, y_test = (
            np.repeat([0, 1], n_averages_train),
            np.repeat([0, 1], n_averages_test),
        )

        # Fit model
        clf.fit(X_train, y_train)

        # Get accuracy and feature importances
        acc_true = acc_true + sklearn.metrics.accuracy_score(
            y_test, clf.predict(X_test)
        )
        fmp_true = fmp_true + clf.feature_importances_

        # Fit null hypothesis model
        clf.fit(X_train, np.random.permutation(y_train))

        # Get null hypothesis accuracy and feature importances
        acc_fake = acc_fake + sklearn.metrics.accuracy_score(
            y_test, clf.predict(X_test)
        )
        fmp_fake = fmp_fake + clf.feature_importances_

    # Scale
    acc_true = acc_true / n_splits
    acc_fake = acc_fake / n_splits
    fmp_true = fmp_true / n_splits
    fmp_fake = fmp_fake / n_splits

    return acc_true, acc_fake, fmp_true, fmp_fake


# Function that calls the classifications
def decode_timeslice(X_all, trialinfo):

    # Set number of trials used for generating an "erp" in the bootstapping procedure
    n_batch = 5

    # Trialinfo cols:
    # 0  id
    # 1  blocknr
    # 2  trialnr
    # 3  bonus
    # 4  tilt_task
    # 5  cue_ax
    # 6  target_red_left
    # 7  distractor_red_left
    # 8  response_interference
    # 9  task_switch
    # 10 correct_response

    # Decode labels
    decode_labels = []

    # Result matrices
    acc_true = np.zeros((5))
    acc_fake = np.zeros((5))
    fmp_true = np.zeros((5, X_all.shape[1]))
    fmp_fake = np.zeros((5, X_all.shape[1]))

    # 1. Decode bonus trials
    decode_labels.append("bonus_trials")
    n_bootstrap_train = 1200
    n_bootstrap_test = 400
    idx_selected = trialinfo[:, 0] > 0  # all trials for now...
    X = X_all[idx_selected, :]
    y = trialinfo[idx_selected, 3]
    (
        acc_true[0],
        acc_fake[0],
        fmp_true[0, :],
        fmp_fake[0, :],
    ) = random_forest_classification(
        X, y, n_batch, n_bootstrap_train, n_bootstrap_test,
    )

    # Exclude incorrect trials
    correct_idx = trialinfo[:, 8] == 1
    X = X[correct_idx, :]
    trialinfo = trialinfo[correct_idx, :]

    # Iterate interruptions and positions
    for irp in range(2):
        for pos in range(2):

            # Trialinfo table columns:
            # id, age, relevant, colpat, fisec, stim, interruption, probert, probeacc, key_pressed, key_correct, probe_interference

            # Get interruption and position specific trials
            intpos_idx = (trialinfo[:, 4] == pos + 1) & (trialinfo[:, 6] == irp)
            X_intpos = X[intpos_idx, :]
            trialinfo_intpos = trialinfo[intpos_idx, :]

            # 1: Decode distractor vs target (coded in trialinfo already as zeros and ones)
            (
                acc_true[0, irp, pos],
                acc_fake[0, irp, pos],
                fmp_true[0, irp, pos, :],
                fmp_fake[0, irp, pos, :],
            ) = random_forest_classification(
                X_intpos,
                trialinfo_intpos[:, 2],
                n_batch,
                n_bootstrap_train,
                n_bootstrap_test,
            )

            # Select target and distractor trial indices
            target_idx = trialinfo_intpos[:, 2] == 1
            distractor_idx = trialinfo_intpos[:, 2] == 0

            # 2: Decode color vresus pattern in targets
            (
                acc_true[1, irp, pos],
                acc_fake[1, irp, pos],
                fmp_true[1, irp, pos, :],
                fmp_fake[1, irp, pos, :],
            ) = random_forest_classification(
                X_intpos[target_idx, :],
                trialinfo_intpos[target_idx, 3] - 1,
                n_batch,
                n_bootstrap_train,
                n_bootstrap_test,
            )

            # 3: Decode color vresus pattern in distractors
            (
                acc_true[2, irp, pos],
                acc_fake[2, irp, pos],
                fmp_true[2, irp, pos, :],
                fmp_fake[2, irp, pos, :],
            ) = random_forest_classification(
                X_intpos[distractor_idx, :],
                trialinfo_intpos[distractor_idx, 3] - 1,
                n_batch,
                n_bootstrap_train,
                n_bootstrap_test,
            )

            # 4: Decode response side in targets
            (
                acc_true[3, irp, pos],
                acc_fake[3, irp, pos],
                fmp_true[3, irp, pos, :],
                fmp_fake[3, irp, pos, :],
            ) = random_forest_classification(
                X_intpos[target_idx, :],
                trialinfo_intpos[target_idx, 9],
                n_batch,
                n_bootstrap_train,
                n_bootstrap_test,
            )

            # 5: Decode response side in distractors
            (
                acc_true[4, irp, pos],
                acc_fake[4, irp, pos],
                fmp_true[4, irp, pos, :],
                fmp_fake[4, irp, pos, :],
            ) = random_forest_classification(
                X_intpos[distractor_idx, :],
                trialinfo_intpos[distractor_idx, 9],
                n_batch,
                n_bootstrap_train,
                n_bootstrap_test,
            )

    return {
        "acc_true": acc_true,
        "acc_fake": acc_fake,
        "fmp_true": fmp_true,
        "fmp_fake": fmp_fake,
    }


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*.set")
list_failed = []
for dataset_idx, dataset in enumerate(datasets):

    # Take time
    tic = time.perf_counter()

    # Talk
    print(f"Decoding dataset {dataset_idx + 1} / {len(datasets)}.")

    # Get subject
    id_string = dataset.split("auto")[0].split("/")[-1][0:4]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Get pruned time vector
    eeg_times = eeg_epochs.times[pruneframes:-pruneframes]

    # Save times
    if dataset_idx == 0:
        out_file = os.path.join(path_out, "eeg_times.joblib")
        joblib.dump(eeg_times, out_file)

    # Load trialinfo
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + id_string + "_trialinfo_tf.csv", delimiter=","
    )

    # Data as numpy arrayas as trials x channels x time
    eeg_data = eeg_epochs.get_data() * 1e6
    n_trials, n_channels, n_times = eeg_data.shape

    # Freqband specific data as numpy arrayas as channels x trials x time
    data_delta = np.zeros((n_channels, n_trials, len(eeg_times)))
    data_theta = np.zeros((n_channels, n_trials, len(eeg_times)))
    data_alpha = np.zeros((n_channels, n_trials, len(eeg_times)))
    data_beta = np.zeros((n_channels, n_trials, len(eeg_times)))
    for channel_idx in range(0, n_channels):

        # Channel data as n_trials x n_times
        data2d = eeg_data[:, channel_idx, :]

        # Concatenate epochs
        data1d = data2d.reshape((n_trials * n_times))

        # Filter
        delta = np.square(np.abs(filter_hilbert(data1d, 2, 3, 200, order=5)))
        theta = np.square(np.abs(filter_hilbert(data1d, 4, 7, 200, order=5)))
        alpha = np.square(np.abs(filter_hilbert(data1d, 8, 12, 200, order=5)))
        beta = np.square(np.abs(filter_hilbert(data1d, 13, 30, 200, order=5)))

        # Back to n_trials x n_times and prune edge artifacts
        data_delta[channel_idx, :, :] = delta.reshape((n_trials, n_times))[
            :, pruneframes:-pruneframes
        ]
        data_theta[channel_idx, :, :] = theta.reshape((n_trials, n_times))[
            :, pruneframes:-pruneframes
        ]
        data_alpha[channel_idx, :, :] = alpha.reshape((n_trials, n_times))[
            :, pruneframes:-pruneframes
        ]
        data_beta[channel_idx, :, :] = beta.reshape((n_trials, n_times))[
            :, pruneframes:-pruneframes
        ]

    # Re-arrange data
    X_list = []
    for time_idx, timeval in enumerate(eeg_times):

        # Data as trials x channels
        delta = data_delta[:, :, time_idx].T
        theta = data_theta[:, :, time_idx].T
        alpha = data_alpha[:, :, time_idx].T
        beta = data_beta[:, :, time_idx].T

        # Stack data
        X_list.append(np.hstack((delta, theta, alpha, beta)))

    # Fit random forest
    out = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(decode_timeslice)(X, trialinfo) for X in X_list
    )

