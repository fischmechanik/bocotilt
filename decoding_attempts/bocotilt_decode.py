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
import imblearn
import scipy.io
import matplotlib.pyplot as plt

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/3_rf_decoding_task_properties_by_reward/"

# Function performing random forest classification
def do_some_classification(X, y):

    # Get dims
    n_trials, n_features = X.shape

    # Init undersampler
    undersampler = imblearn.under_sampling.RandomUnderSampler(
        sampling_strategy="not minority"
    )
    
    # Init classifier
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    
    # Set number of iterations
    n_iterations = 20
    
    # List for classifier performance and feature importances
    acc = []
    fmp = []
    
    # Loop iterations
    for iteration_idx in range(n_iterations):
        
        # Undersample data
        X_undersampled, y_undersampled = undersampler.fit_resample(X, y)
        
        # Shuffle data after undersampling
        X_undersampled, y_undersampled = sklearn.utils.shuffle(X_undersampled, y_undersampled)
        
        # Get data for both classes
        X0 = X_undersampled[y_undersampled == 0, :]
        X1 = X_undersampled[y_undersampled == 1, :]
        
        # Set binsize
        binsize = 10
        
        # Determine number of bins
        n_bins = int(np.floor(X0.shape[0] / binsize))
        
        # Arrays for bins
        X_binned_0 = np.zeros((n_bins, n_features))
        X_binned_1 = np.zeros((n_bins, n_features))
        
        # Binning. Create ERPs
        for row_idx, X_idx in enumerate(np.arange(0, X0.shape[0], 10)[:-1]):
            X_binned_0[row_idx, :] = X0[X_idx : X_idx + 10, :].mean(axis=0)
            X_binned_1[row_idx, :] = X1[X_idx : X_idx + 10, :].mean(axis=0)
            
        # Concatenate bins
        X_binned = np.concatenate((X_binned_0, X_binned_1), axis=0)
        y_binned = np.concatenate((np.zeros((n_bins,)), np.ones((n_bins,))), axis=0)
        
        # Shuffle data after bin creation
        X_binned, y_binned = sklearn.utils.shuffle(X_binned, y_binned)
        
        # Iterate bins
        for bin_idx in range(n_bins):
            
            # Test data
            X_test = X_binned[bin_idx, :].reshape(1, -1)
            y_test = y_binned[bin_idx].reshape(1, -1)
            
            # Train data
            X_train = np.delete(X_binned, bin_idx, 0)
            y_train = np.delete(y_binned, bin_idx, 0)
            
            # Fit model
            clf.fit(X_train, y_train)
    
            # Get accuracy
            acc.append(sklearn.metrics.accuracy_score(y_test, clf.predict(X_test)))
            
            # Get feature importances
            fmp.append(clf.feature_importances_)

    # Average
    average_acc = np.stack(acc).mean(axis=0)
    average_fmp = np.stack(fmp).mean(axis=0)

    return average_acc, average_fmp


# Function that calls the classifications
def decode_timeslice(X_all, trialinfo, prog):
    
    # Talk about making progress
    print(f"decoding dataset {prog[0]}/{prog[1]} - timeslice {prog[2]}/{prog[3]}")

    # Trialinfo cols:
    # 00: id
    # 01: block_nr
    # 02: trial_nr
    # 03: bonustrial
    # 04: tilt_task
    # 05: cue_ax
    # 06: target_red_left
    # 07: distractor_red_left
    # 08: response_interference
    # 09: task_switch
    # 10: prev_switch
    # 11: prev_accuracy
    # 12: correct_response
    # 13: response_side
    # 14: rt
    # 15: rt_thresh_color
    # 16: rt_thresh_tilt
    # 17: accuracy
    # 18: position_color
    # 19: position_tilt
    # 20: position_target
    # 21: position_distractor
    # 22: sequence_position

    # Define classifications to perform
    clfs = []

    # Bonus decoding
    clfs.append(
        {
            "label": "bonus vs standard trials in tilt",
            "trial_idx": trialinfo[:, 0] != 1000,
            "y_col": 3,
        }
    )
    
    # Task decoding
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

    # Cue decoding
    clfs.append(
        {
            "label": "task cue in standard trials in color",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 0),
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "task cue in standard trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 1),
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "task cue in bonus trials in color",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 0),
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "task cue in bonus trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 1),
            "y_col": 5,
        }
    )

    # Response decoding
    clfs.append(
        {
            "label": "Response in standard trials in color",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 0),
            "y_col": 13,
        }
    )
    clfs.append(
        {
            "label": "Response in standard trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 1),
            "y_col": 13,
        }
    )
    clfs.append(
        {
            "label": "Response in bonus trials in color",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 0),
            "y_col": 13,
        }
    )
    clfs.append(
        {
            "label": "Response in bonus trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 1),
            "y_col": 13,
        }
    )

    # Target decoding
    clfs.append(
        {
            "label": "Target in standard trials in color",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 0),
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "Target in standard trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 1),
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "Target in bonus trials in color",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 0),
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "Target in bonus trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 1),
            "y_col": 20,
        }
    )

    # Distractor decoding
    clfs.append(
        {
            "label": "Distractor in standard trials in color",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 0),
            "y_col": 21,
        }
    )
    clfs.append(
        {
            "label": "Distractor in standard trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 0) & (trialinfo[:, 4] == 1),
            "y_col": 21,
        }
    )
    clfs.append(
        {
            "label": "Distractor in bonus trials in color",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 0),
            "y_col": 21,
        }
    )
    clfs.append(
        {
            "label": "Distractor in bonus trials in tilt",
            "trial_idx": (trialinfo[:, 3] == 1) & (trialinfo[:, 4] == 1),
            "y_col": 21,
        }
    )
 
    # Decode labels
    decode_labels = []

    # Result matrices
    acc = np.zeros((len(clfs)))
    fmp = np.zeros((len(clfs), X_all.shape[1]))

    # Perform classifications
    for model_nr, clf in enumerate(clfs):

        # Append classification label
        decode_labels.append(clf["label"])

        # Select features and labels
        X = X_all[clf["trial_idx"], :]
        y = trialinfo[clf["trial_idx"], clf["y_col"]]

        # Train model
        (
            acc[model_nr],
            fmp[model_nr, :],
        ) = do_some_classification(X, y)

    return {
        "decode_labels": decode_labels,
        "acc": acc,
        "fmp": fmp,
    }


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Talk
    print(f"Decoding dataset {dataset_idx + 1} / {len(datasets)}.")

    # Set fs
    srate = 200

    # Read channel labels as list
    channel_label_list = scipy.io.loadmat(os.path.join(path_in, "channel_labels.mat"))[
        "channel_labels"
    ][0].split(" ")[1:]

    # Load epoch data
    eeg_data = scipy.io.loadmat(dataset)["data"].transpose((2, 0, 1))

    # Load epoch times
    eeg_times = scipy.io.loadmat(dataset)["times"][0]

    # Create info struct
    eeg_info = mne.create_info(channel_label_list, srate)

    # Create epoch struct
    eeg_epochs = mne.EpochsArray(eeg_data, eeg_info, tmin=-1)

    # Create channel type mapping
    mapping = {}
    for x in channel_label_list:
        mapping[x] = "eeg"

    # Apply mapping
    eeg_epochs.set_channel_types(mapping)

    # Set montage
    montage = mne.channels.make_standard_montage("standard_1005").rename_channels(
        {"OI1": "O9", "OI2": "O10"}
    )
    eeg_epochs.set_montage(montage)

    # Load trialinfo
    trialinfo = scipy.io.loadmat(dataset)["trialinfo"]

    # Perform single trial time-frequency analysis
    n_freqs = 20
    tf_freqs = np.linspace(2, 30, n_freqs)
    tf_cycles = np.linspace(3, 12, n_freqs)
    tf_epochs = mne.time_frequency.tfr_morlet(
        eeg_epochs,
        tf_freqs,
        n_cycles=tf_cycles,
        picks=np.arange(0, 127),
        average=False,
        return_itc=False,
        n_jobs=-2,
        decim=4,
    )

    # Apply baseline procedure
    # tf_epochs.apply_baseline(mode="logratio", baseline=(-0.100, 0))

    # Prune in time
    to_keep_idx = (tf_epochs.times >= -0.6) & (tf_epochs.times <= 1.6)
    tf_times = tf_epochs.times[to_keep_idx]
    tf_data = tf_epochs.data[:, :, :, to_keep_idx]

    # Clean up
    del eeg_epochs, tf_epochs, eeg_data

    # Positions of target and distractor are coded  1-8, starting at the top-right position, then counting counter-clockwise

    # Recode distractor and target positions in 4 bins 0-3 (c.f. https://www.nature.com/articles/s41598-019-45333-6)
    # trialinfo[:, 19] = np.floor((trialinfo[:, 19] - 1) / 2)
    # trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 2)

    # Recode distractor and target positions in 2 bins 0-1 (roughly left vs right...)
    trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 4)
    trialinfo[:, 21] = np.floor((trialinfo[:, 21] - 1) / 4)

    # Exclude trials: Practice-block trials and first-of-sequence trials and no-response trials
    idx_to_keep = (trialinfo[:, 1] >= 5) & (trialinfo[:, 22] > 1) & ((trialinfo[:, 13] > -1) & (trialinfo[:, 13] < 2))
    trialinfo = trialinfo[idx_to_keep, :]
    tf_data = tf_data[idx_to_keep, :, :, :]

    # get dims
    n_trials, n_channels, n_freqs, n_times = tf_data.shape

    # Trialinfo cols:
    # 00: id
    # 01: block_nr
    # 02: trial_nr
    # 03: bonustrial
    # 04: tilt_task
    # 05: cue_ax
    # 06: target_red_left
    # 07: distractor_red_left
    # 08: response_interference
    # 09: task_switch
    # 10: prev_switch
    # 11: prev_accuracy
    # 12: correct_response
    # 13: response_side
    # 14: rt
    # 15: rt_thresh_color
    # 16: rt_thresh_tilt
    # 17: accuracy
    # 18: position_color
    # 19: position_tilt
    # 20: position_target
    # 21: position_distractor
    # 22: sequence_position

    # Re-arrange data
    X_list = []
    temporal_smoothing = 2
    tf_times = tf_times[:-(temporal_smoothing-1)]
    for time_idx, timeval in enumerate(tf_times):

        # Data as trials x channels x frequencies. Apply a temporal smoothing
        timepoint_data = tf_data[:, :, :, time_idx : time_idx + temporal_smoothing].mean(axis=3)

        # Trials in rows
        timepoint_data_2d = timepoint_data.reshape((n_trials, n_channels * n_freqs))

        # Stack data
        X_list.append(timepoint_data_2d)

    # Clean up
    del tf_data

    # Fit random forest
    out = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(decode_timeslice)(X, trialinfo, (dataset_idx, len(datasets), X_idx, len(X_list))) for X_idx, X in enumerate(X_list)
    )

    # Re-arrange data into arrays
    acc = np.stack([x["acc"] for x in out])
    fmp = np.stack([x["fmp"] for x in out])

    plt.plot(tf_times, acc)
    
    # Get number of classifications performed
    n_clf = acc.shape[1]

    # Reshape feature space data
    fmp = fmp.reshape((n_times, n_clf, n_channels, n_freqs))

    # Get decode labels
    decode_labels = out[0]["decode_labels"]

    # Re-arrange decoding-results as classification-specific lists
    acc = [acc[:, i] for i in range(n_clf)]
    fmp = [fmp[:, i, :, :] for i in range(n_clf)]

    # Compile output
    output = {
        "decode_labels": decode_labels,
        "tf_times": tf_times,
        "tf_freqs": tf_freqs,
        "acc": acc,
        "fmp": fmp,
    }

    # Specify out file name
    out_file = os.path.join(path_out, f"{id_string}_decoding_data.joblib")

    # Save
    joblib.dump(output, out_file)


