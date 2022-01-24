#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import os
import joblib
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import scipy.signal
import scipy.io
import mne
import time
import imblearn

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
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
def random_forest_classification(X, y, combined_codes):

    # Condition differences in frequency x channel space
    cond_dif = X[y == 0, :].mean(axis=0) - X[y == 1, :].mean(axis=0)

    # Test statistics in frequency x channel space
    t_values, p_values = scipy.stats.ttest_ind(X[y == 0, :], X[y == 1, :], axis=0)

    # Calculate adjusted partial eta squared in frequency x channel space
    df = np.min((X[y == 0, :].shape[0] - 1, X[y == 1, :].shape[0] - 1))
    petasq = np.divide(np.square(t_values), (np.square(t_values) + df))
    adjpetsq = petasq - np.multiply(1 - petasq, 1 / df)

    # Get dims
    n_trials, n_channel = X.shape

    # Oversampler
    oversampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy="minority")

    # Init splitter
    n_splits = 5
    kf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)

    # Init classifier
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
        random_state=42,
    )

    # Arrays for results
    acc_true = 0
    fmp_true = np.zeros((n_channel))
    acc_fake = 0
    fmp_fake = np.zeros((n_channel))

    # Iterate splits
    for kidx, (idx_train, idx_test) in enumerate(kf.split(X, y)):

        # Split data
        X_train, X_test = X[idx_train, :], X[idx_test, :]
        y_train, y_test = y[idx_train], y[idx_test]
        cc_train = combined_codes[idx_train]
        
        # Combine X and y for combined oversampling
        Xy_train = np.hstack((X_train, y_train.reshape(-1,1)))

        # Oversample training data
        Xy_train, cc_oversampled = oversampler.fit_resample(Xy_train, cc_train)

        # Split X and y again
        X_train, y_train = Xy_train[:, : -1], np.squeeze(Xy_train[:, -1 :])
    
        # Shuffle 
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
    
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

    return acc_true, acc_fake, fmp_true, fmp_fake, cond_dif, t_values, adjpetsq


# Function that calls the classifications
def decode_timeslice(X_all, trialinfo, combined_codes):

    # Trialinfo cols:
    #  0: id
    #  1: block_nr
    #  2: trial_nr
    #  3: bonustrial  x
    #  4: tilt_task  x
    #  5: cue_ax  x
    #  6: target_red_left  x
    #  7: distractor_red_left  x
    #  8: response_interference  x
    #  9: task_switch  x
    # 10: correct_response x
    # 11: response_side
    # 12: rt
    # 13: accuracy
    # 14: log_response_side
    # 15: log_rt
    # 16: log_accuracy
    # 17: position_color  x
    # 18: position_tilt  x
    # 19: position_target   x
    # 20: position_distractor   x
    # 21: sequence_position
    # 22: sequence_length

    # Decode labels
    decode_labels = []

    # Result matrices
    n_models = 2
    acc_true = np.zeros((n_models))
    acc_fake = np.zeros((n_models))
    fmp_true = np.zeros((n_models, X_all.shape[1]))
    fmp_fake = np.zeros((n_models, X_all.shape[1]))
    cond_dif = np.zeros((n_models, X_all.shape[1]))
    t_values = np.zeros((n_models, X_all.shape[1]))
    adjpetsq = np.zeros((n_models, X_all.shape[1]))

    # 0. Decode standard vs bonus trials
    model_nr = 0
    decode_labels.append("bonus vs standard trials")
    idx_selected = trialinfo[:, 0] > 0  # all trials for now...
    X = X_all[idx_selected, :]
    y = trialinfo[idx_selected, 3]
    (
        acc_true[model_nr],
        acc_fake[model_nr],
        fmp_true[model_nr, :],
        fmp_fake[model_nr, :],
        cond_dif[model_nr, :],
        t_values[model_nr, :],
        adjpetsq[model_nr, :],
    ) = random_forest_classification(
        X, y, combined_codes,
    )



    return {
        "decode_labels": decode_labels,
        "acc_true": acc_true,
        "acc_fake": acc_fake,
        "fmp_true": fmp_true,
        "fmp_fake": fmp_fake,
        "cond_dif": cond_dif,
        "t_values": t_values,
        "adjpetsq": adjpetsq,
    }


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")
list_failed = []
for dataset_idx, dataset in enumerate(datasets):

    # Take time
    tic = time.perf_counter()

    # Talk
    print(f"Decoding dataset {dataset_idx + 1} / {len(datasets)}.")

    # Get subject
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Get pruned time vector
    eeg_times = eeg_epochs.times[pruneframes:-pruneframes]

    # Load trialinfo
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )
    
    # Data as numpy arrayas as trials x channels x time
    eeg_data = eeg_epochs.get_data() * 1e6

    # Exclude non-valid switch-repetition trials and practice blocks
    idx_to_keep = (trialinfo[:, 9] >= 0) & (trialinfo[:, 1] >= 5)
    trialinfo = trialinfo[idx_to_keep, :]
    eeg_data = eeg_data[idx_to_keep, :, :]
    
    # get dims
    n_trials, n_channels, n_times = eeg_data.shape
    
    # Trialinfo cols:
    #  0: id
    #  1: block_nr
    #  2: trial_nr
    #  3: bonustrial  x
    #  4: tilt_task  x
    #  5: cue_ax  x
    #  6: target_red_left  x
    #  7: distractor_red_left  x
    #  8: response_interference  x
    #  9: task_switch  x
    # 10: correct_response x
    # 11: response_side
    # 12: rt
    # 13: accuracy
    # 14: log_response_side
    # 15: log_rt
    # 16: log_accuracy
    # 17: position_color  x
    # 18: position_tilt  x
    # 19: position_target   x
    # 20: position_distractor   x
    # 21: sequence_position
    # 22: sequence_length

    # Get combined coding
    combined_codes = np.array(
        [
            int(
                str(int(trialinfo[x, 3]))
                + str(int(trialinfo[x, 4]))
                + str(int(trialinfo[x, 5]))
                + str(int(trialinfo[x, 6]))
                + str(int(trialinfo[x, 7]))
                + str(int(trialinfo[x, 8]))
                + str(int(trialinfo[x, 9]))
                + str(int(trialinfo[x, 10]))
                + str(int(trialinfo[x, 17]))
                + str(int(trialinfo[x, 18]))
                + str(int(trialinfo[x, 19]))
                + str(int(trialinfo[x, 20]))
            )
            for x in range(trialinfo.shape[0])
        ]
    )

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
        joblib.delayed(decode_timeslice)(X, trialinfo, combined_codes) for X in X_list
    )

    # Re-arrange data into arrays
    acc_true = np.stack([x["acc_true"] for x in out])
    acc_fake = np.stack([x["acc_fake"] for x in out])
    fmp_true = np.stack([x["fmp_true"] for x in out])
    fmp_fake = np.stack([x["fmp_fake"] for x in out])

    # Get number of classifications performed
    n_clf = acc_true.shape[1]

    # Get decode labels
    decode_labels = out[0]["decode_labels"]

    # Split feature importances into freqbands
    fmp_true = np.split(fmp_true, 4, axis=2)
    fmp_fake = np.split(fmp_fake, 4, axis=2)
    fmp_true_delta = fmp_true[0]
    fmp_fake_delta = fmp_fake[0]
    fmp_true_theta = fmp_true[1]
    fmp_fake_theta = fmp_fake[1]
    fmp_true_alpha = fmp_true[2]
    fmp_fake_alpha = fmp_fake[2]
    fmp_true_beta = fmp_true[3]
    fmp_fake_beta = fmp_fake[3]

    # Re-arrange decoding-results as classification-specific lists
    acc_true = [acc_true[:, i] for i in range(n_clf)]
    acc_fake = [acc_fake[:, i] for i in range(n_clf)]
    fmp_true_delta = [fmp_true_delta[:, i, :] for i in range(n_clf)]
    fmp_fake_delta = [fmp_fake_delta[:, i, :] for i in range(n_clf)]
    fmp_true_theta = [fmp_true_theta[:, i, :] for i in range(n_clf)]
    fmp_fake_theta = [fmp_fake_theta[:, i, :] for i in range(n_clf)]
    fmp_true_alpha = [fmp_true_alpha[:, i, :] for i in range(n_clf)]
    fmp_fake_alpha = [fmp_fake_alpha[:, i, :] for i in range(n_clf)]
    fmp_true_beta = [fmp_true_beta[:, i, :] for i in range(n_clf)]
    fmp_fake_beta = [fmp_fake_beta[:, i, :] for i in range(n_clf)]

    # Compile output
    output = {
        "decode_labels": decode_labels,
        "eeg_times": eeg_times,
        "acc_true": acc_true,
        "acc_fake": acc_fake,
        "fmp_true_delta": fmp_true_delta,
        "fmp_fake_delta": fmp_fake_delta,
        "fmp_true_theta": fmp_true_theta,
        "fmp_fake_theta": fmp_fake_theta,
        "fmp_true_alpha": fmp_true_alpha,
        "fmp_fake_alpha": fmp_fake_alpha,
        "fmp_true_beta": fmp_true_beta,
        "fmp_fake_beta": fmp_fake_beta,
    }

    # Save
    out_file = os.path.join(path_out, f"{id_string}_decoding_data.joblib")
    joblib.dump(output, out_file)

    # Take time
    toc = time.perf_counter()

    # Talk again
    print(f"dataset completed in {toc - tic:0.4f} seconds")

