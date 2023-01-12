#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import os
import joblib
import numpy as np
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.decomposition
import mne
import time
import imblearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/3_decoded/nn_test/"

# Function performing random forest classification
def nn_classification(time_idx, X, y, combined_codes):

    # Oversampler
    oversampler = imblearn.over_sampling.RandomOverSampler(
        sampling_strategy="not majority"
    )

    # Init splitter
    n_splits = 3
    kf = sklearn.model_selection.StratifiedKFold(n_splits=n_splits)

    # Arrays for results
    acc_classy = 0

    # Iterate splits
    for kidx, (idx_train, idx_test) in enumerate(kf.split(X, y)):
        
        # Detect number of classes
        n_classes = len(np.unique(y))

        # Split data
        X_train, X_test = X[idx_train, :], X[idx_test, :]
        y_train, y_test = y[idx_train], y[idx_test]
        cc_train = combined_codes[idx_train]

        # Combine X and y for combined oversampling
        Xy_train = np.hstack((X_train, y_train.reshape(-1, 1)))

        # Oversample training data
        Xy_train, cc_oversampled = oversampler.fit_resample(Xy_train, cc_train)

        # Split X and y again
        X_train_full, y_train_full = Xy_train[:, :-1], np.squeeze(Xy_train[:, -1:])

        # Shuffle training data after oversampling
        X_train_full, y_train_full = sklearn.utils.shuffle(X_train_full, y_train_full)

        # Split training data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.2
        )
        
        # Specify neural-network via keras functional API
        n_layer = 2
        n_neurons = 20
        activation_function = "elu"
        initializer = "he_normal"
        input_layer = keras.layers.Input(shape=X_train.shape[1], name="input")
        hidden_layers = []
        for hl_idx in range(n_layer):
            if hl_idx == 0:
                inlay = input_layer
            else:
                inlay = hidden_layers[-1]
            hidden_layers.append(
                keras.layers.Dense(
                    n_neurons,
                    activation=activation_function,
                    kernel_initializer=initializer,
                    name=f"hidden_{hl_idx}",
                )(inlay)
            )
        output_layer = keras.layers.Dense(n_classes, activation="softmax", name="output")(
            hidden_layers[-1]
        )

        # Compile the model
        model = keras.Model(inputs=[input_layer], outputs=[output_layer])
        model.summary()
        model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy())

        # Define callbacks
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            f"{path_out}callback_time_{time_idx}.h5",
            save_best_only=True,
        )
        patience = 10
        earlystop_cb = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=patience
        )

        # Fit the model
        history = model.fit(
            X_train,
            y_train,
            epochs=1000,
            validation_data=([X_valid], [y_valid]),
            callbacks=[checkpoint_cb, earlystop_cb],
            batch_size=32,
        )

        # Load best model
        model = tf.keras.models.load_model(f"{path_out}callback_time_{time_idx}.h5")

        # Evaluate model on test data
        acc_classy = acc_classy + model.evaluate(X_test, y_test, verbose=0)

    # Scale
    acc_classy = acc_classy / n_splits

    return acc_classy


# Function that calls the classifications
def decode_timeslice(time_idx, X_all, trialinfo, combined_codes):

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
    #  9: task_switch x
    # 10: prev_switch x
    # 11: prev_accuracy
    # 12: correct_response
    # 13: response_side x
    # 14: rt
    # 15: rt_thresh_color
    # 16: rt_thresh_tilt
    # 17: accuracy
    # 18: position_color
    # 19: position_tilt
    # 20: position_target   x
    # 21: position_distractor   x
    # 22: sequence_position

    # Define classifications to perform
    clfs = []

    # Get condition indices
    idx_std_rep = (trialinfo[:, 3] == 0) & (trialinfo[:, 9] == 0)
    idx_std_swi = (trialinfo[:, 3] == 0) & (trialinfo[:, 9] == 1)
    idx_bon_rep = (trialinfo[:, 3] == 1) & (trialinfo[:, 9] == 0)
    idx_bon_swi = (trialinfo[:, 3] == 1) & (trialinfo[:, 9] == 1)

    # Task decoding
    clfs.append(
        {
            "label": "task in std-rep",
            "trial_idx": idx_std_rep,
            "y_col": 4,
        }
    )

    clfs.append(
        {
            "label": "task in std-swi",
            "trial_idx": idx_std_swi,
            "y_col": 4,
        }
    )
    clfs.append(
        {
            "label": "task in bon-rep",
            "trial_idx": idx_bon_rep,
            "y_col": 4,
        }
    )

    clfs.append(
        {
            "label": "task in bon-swi",
            "trial_idx": idx_bon_swi,
            "y_col": 4,
        }
    )

    # Cue decoding
    clfs.append(
        {
            "label": "cue in std-rep",
            "trial_idx": idx_std_rep,
            "y_col": 5,
        }
    )

    clfs.append(
        {
            "label": "cue in std-swi",
            "trial_idx": idx_std_swi,
            "y_col": 5,
        }
    )
    clfs.append(
        {
            "label": "cue in bon-rep",
            "trial_idx": idx_bon_rep,
            "y_col": 5,
        }
    )

    clfs.append(
        {
            "label": "cue in bon-swi",
            "trial_idx": idx_bon_swi,
            "y_col": 5,
        }
    )

    # Response decoding
    clfs.append(
        {
            "label": "response in std-rep",
            "trial_idx": idx_std_rep,
            "y_col": 13,
        }
    )

    clfs.append(
        {
            "label": "response in std-swi",
            "trial_idx": idx_std_swi,
            "y_col": 13,
        }
    )
    clfs.append(
        {
            "label": "response in bon-rep",
            "trial_idx": idx_bon_rep,
            "y_col": 13,
        }
    )

    clfs.append(
        {
            "label": "response in bon-swi",
            "trial_idx": idx_bon_swi,
            "y_col": 13,
        }
    )

    # Target decoding
    clfs.append(
        {
            "label": "target in std-rep",
            "trial_idx": idx_std_rep,
            "y_col": 20,
        }
    )

    clfs.append(
        {
            "label": "target in std-swi",
            "trial_idx": idx_std_swi,
            "y_col": 20,
        }
    )
    clfs.append(
        {
            "label": "target in bon-rep",
            "trial_idx": idx_bon_rep,
            "y_col": 20,
        }
    )

    clfs.append(
        {
            "label": "target in bon-swi",
            "trial_idx": idx_bon_swi,
            "y_col": 20,
        }
    )

    # Distractor decoding
    clfs.append(
        {
            "label": "distractor in std-rep",
            "trial_idx": idx_std_rep,
            "y_col": 21,
        }
    )

    clfs.append(
        {
            "label": "distractor in std-swi",
            "trial_idx": idx_std_swi,
            "y_col": 21,
        }
    )
    clfs.append(
        {
            "label": "distractor in bon-rep",
            "trial_idx": idx_bon_rep,
            "y_col": 21,
        }
    )

    clfs.append(
        {
            "label": "distractor in bon-swi",
            "trial_idx": idx_bon_swi,
            "y_col": 21,
        }
    )

    # Decode labels
    decode_labels = []

    # Result matrices
    acc = np.zeros((len(clfs)))

    # Perform classifications
    for model_nr, clf in enumerate(clfs):

        # Append classification label
        decode_labels.append(clf["label"])

        # Select features and labels
        X = X_all[clf["trial_idx"], :]
        y = trialinfo[clf["trial_idx"], clf["y_col"]]

        # Train model
        acc[model_nr] = nn_classification(time_idx, X, y, combined_codes)

    return {
        "decode_labels": decode_labels,
        "acc": acc,
    }


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Specify out file name
    fn_out = f"{id_string}_decoding_data_logistic_regression.joblib"
    out_file = os.path.join(path_out, fn_out)

    # Skip if file exists already
    out_file = os.path.join(path_out, fn_out)
    if os.path.isfile(out_file):
        continue

    # Take time
    tic = time.perf_counter()

    # Talk
    print(f"Decoding dataset {dataset_idx + 1} / {len(datasets)}.")

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Perform single trial time-frequency analysis
    tf_freqs = np.linspace(2, 31, 10)
    tf_cycles = np.linspace(3, 14, 10)
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
    tf_epochs.crop(tmin=-0.2, tmax=1.6)

    # Load trialinfo
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )

    # Positions of target and distractor are coded  1-8, starting at the top-right position, then counting counter-clockwise

    # Recode distractor and target positions in 4 bins 0-3 (c.f. https://www.nature.com/articles/s41598-019-45333-6)
    trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 2)
    trialinfo[:, 21] = np.floor((trialinfo[:, 21] - 1) / 2)

    # Recode distractor and target positions in 2 bins 0-1 (roughly left vs right...)
    # trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 4)
    # trialinfo[:, 21] = np.floor((trialinfo[:, 21] - 1) / 4)

    # Exclude trials: Practice-block trials & first-of-sequence trials & no response trials
    idx_to_keep = (
        (trialinfo[:, 1] >= 5) & (trialinfo[:, 22] >= 2) & (trialinfo[:, 13] != 2)
    )
    trialinfo = trialinfo[idx_to_keep, :]
    tf_data = tf_epochs.data[idx_to_keep, :, :, :]
    tf_times = tf_epochs.times

    # Clean up
    del eeg_epochs
    del tf_epochs

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
    #  9: task_switch x
    # 10: prev_switch x
    # 11: prev_accuracy
    # 12: correct_response
    # 13: response_side x
    # 14: rt
    # 15: rt_thresh_color
    # 16: rt_thresh_tilt
    # 17: accuracy
    # 18: position_color
    # 19: position_tilt
    # 20: position_target   x
    # 21: position_distractor   x
    # 22: sequence_position

    # Get combined coding
    combined_codes = np.array(
        [
            int(
                str(int(trialinfo[x, 3]))
                + str(int(trialinfo[x, 4]))
                + str(int(trialinfo[x, 5]))
                + str(int(trialinfo[x, 9]))
                + str(int(trialinfo[x, 10]))
                + str(int(trialinfo[x, 20]))
                + str(int(trialinfo[x, 21]))
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

    # Decode
    out = joblib.Parallel(n_jobs=-2)(
        joblib.delayed(decode_timeslice)(time_idx, X, trialinfo, combined_codes)
        for time_idx, X in enumerate(X_list)
    )

    # Re-arrange data into arrays
    acc = np.stack([x["acc"] for x in out])

    # Get number of classifications performed
    n_clf = acc.shape[1]

    # Get decode labels
    decode_labels = out[0]["decode_labels"]

    # Re-arrange decoding-results as classification-specific lists
    acc = [acc[:, i] for i in range(n_clf)]

    # Compile output
    output = {
        "decode_labels": decode_labels,
        "tf_times": tf_times,
        "acc": acc,
    }

    # Save
    joblib.dump(output, out_file)

    # Take time
    toc = time.perf_counter()

    # Talk again
    print(f"dataset completed in {toc - tic:0.4f} seconds")
