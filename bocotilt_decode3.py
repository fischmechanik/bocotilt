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
import mne
import time
import imblearn

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/3_decoded/"

# Function performing random forest classification
def random_forest_classification(X, y, combined_codes):

    # Get dims
    n_trials, n_channel = X.shape

    # Oversampler
    oversampler = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy="not majority"
    )

    # Init splitter
    n_splits = 3
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
        Xy_train = np.hstack((X_train, y_train.reshape(-1, 1)))

        # Oversample training data
        Xy_train, cc_oversampled = oversampler.fit_resample(Xy_train, cc_train)

        # Split X and y again
        X_train, y_train = Xy_train[:, :-1], np.squeeze(Xy_train[:, -1:])

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

    return acc_true, acc_fake, fmp_true, fmp_fake


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

    # Define classifications to perform
    clfs = []
    clfs.append(
        {
            "label": "bonus vs standard trials",
            "trial_idx": trialinfo[:, 0] > 0,
            "y_col": 3,
        }
    )
    clfs.append(
        {
            "label": "task in standard trials",
            "trial_idx": trialinfo[:, 3] == 0,
            "y_col": 4,
        }
    )
    clfs.append(
        {
            "label": "task in bonus trials",
            "trial_idx": trialinfo[:, 3] == 1,
            "y_col": 4,
        }
    )
    clfs.append(
        {
            "label": "task cue in standard trials",
            "trial_idx": trialinfo[:, 3] == 0,
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "task cue in bonus trials",
            "trial_idx": trialinfo[:, 3] == 1,
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "target in standard trials",
            "trial_idx": trialinfo[:, 3] == 0,
            "y_col": 19,
        }
    )
    clfs.append(
        {
            "label": "target in bonus trials",
            "trial_idx": trialinfo[:, 3] == 1,
            "y_col": 19,
        }
    )
    clfs.append(
        {
            "label": "distractor in standard trials",
            "trial_idx": trialinfo[:, 3] == 0,
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "distractor in bonus trials",
            "trial_idx": trialinfo[:, 3] == 1,
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "response in standard trials",
            "trial_idx": trialinfo[:, 3] == 0,
            "y_col": 14,
        }
    )
    clfs.append(
        {
            "label": "response in bonus trials",
            "trial_idx": trialinfo[:, 3] == 1,
            "y_col": 14,
        }
    )

    # Decode labels
    decode_labels = []

    # Result matrices
    acc_true = np.zeros((len(clfs)))
    acc_fake = np.zeros((len(clfs)))
    fmp_true = np.zeros((len(clfs), X_all.shape[1]))
    fmp_fake = np.zeros((len(clfs), X_all.shape[1]))

    # Perform classifications
    for model_nr, clf in enumerate(clfs):
        
        # Select features and labels
        X = X_all[clf["trial_idx"], :]
        y = trialinfo[clf["trial_idx"], clf["y_col"]]

        # Count occurences because why not...
        unique, counts = np.unique(y, return_counts=True)
        print(
            f"clf: '{clf['label']}' | n obs: {dict(zip(unique.astype('int'), counts))}"
        )

        # Train model 
        (
            acc_true[model_nr],
            acc_fake[model_nr],
            fmp_true[model_nr, :],
            fmp_fake[model_nr, :],
        ) = random_forest_classification(X, y, combined_codes)

    return {
        "decode_labels": decode_labels,
        "acc_true": acc_true,
        "acc_fake": acc_fake,
        "fmp_true": fmp_true,
        "fmp_fake": fmp_fake,
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

    # Perform single trial time-frequency analysis
    tf_freqs = np.arange(2, 20)
    tf_epochs = mne.time_frequency.tfr_morlet(
        eeg_epochs,
        tf_freqs,
        n_cycles=4.0,
        average=False,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )

    # Apply baseline procedure
    tf_epochs.apply_baseline(mode="logratio", baseline=(-0.100, 0))
    
    # Clean up
    del eeg_epochs

    # Prune in time
    pruneframes = 60
    tf_times = tf_epochs.times[pruneframes:-pruneframes]
    tf_data = tf_epochs.data[:, :, :, pruneframes:-pruneframes]
    
    # Clean up
    del tf_epochs

    # Load trialinfo
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )

    # Bin distractor and target positions (4 bins, c.f. https://www.nature.com/articles/s41598-019-45333-6)
    trialinfo[:, 19] = np.floor((trialinfo[:, 19] - 1) / 2)
    trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 2)

    # Exclude non-valid switch-repetition trials and practice blocks
    idx_to_keep = (trialinfo[:, 9] >= 0) & (trialinfo[:, 1] >= 5)
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
    #  6: target_red_left  x
    #  7: distractor_red_left  x
    #  8: response_interference  x
    #  9: task_switch  x
    # 10: correct_response
    # 11: response_side
    # 12: rt
    # 13: accuracy
    # 14: log_response_side
    # 15: log_rt
    # 16: log_accuracy
    # 17: position_color
    # 18: position_tilt
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
                + str(int(trialinfo[x, 19]))
                + str(int(trialinfo[x, 20]))
            )
            for x in range(trialinfo.shape[0])
        ]
    )

    # Re-arrange data
    X_list = []
    for time_idx, timeval in enumerate(tf_times):

        # Data as trials x channels x frequencies
        timepoint_data = tf_data[:, :, :, time_idx]

        # Trials in rows
        timepoint_data_2d = timepoint_data.reshape((n_trials, n_channels * n_freqs))

        # Stack data
        X_list.append(timepoint_data_2d)
        
    # Clean up
    del tf_data

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

    # Reshape feature space data
    fmp_true = fmp_true.reshape((n_times, n_clf, n_channels, n_freqs))
    fmp_fake = fmp_fake.reshape((n_times, n_clf, n_channels, n_freqs))

    # Get decode labels
    decode_labels = out[0]["decode_labels"]

    # Re-arrange decoding-results as classification-specific lists
    acc_true = [acc_true[:, i] for i in range(n_clf)]
    acc_fake = [acc_fake[:, i] for i in range(n_clf)]
    fmp_true = [fmp_true[:, i, :, :] for i in range(n_clf)]
    fmp_fake = [fmp_fake[:, i, :, :] for i in range(n_clf)]

    # Compile output
    output = {
        "decode_labels": decode_labels,
        "tf_times": tf_times,
        "tf_freqs": tf_freqs,
        "acc_true": acc_true,
        "acc_fake": acc_fake,
        "fmp_true": fmp_true,
        "fmp_fake": fmp_fake,
    }

    # Save
    out_file = os.path.join(path_out, f"{id_string}_decoding_data.joblib")
    joblib.dump(output, out_file)

    # Take time
    toc = time.perf_counter()

    # Talk again
    print(f"dataset completed in {toc - tic:0.4f} seconds")

