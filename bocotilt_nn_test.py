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

# Function for bootstrapping averages
def do_some_binning(data_in, n_batch, n_averages):
    averages = np.zeros((n_averages, data_in.shape[1]))
    for n in range(n_averages):
        averages[n, :] = data_in[np.random.choice(data_in.shape[0], n_batch), :].mean(
            axis=0
        )
    return averages

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/3_decoded/nn_test/"

# Load a dataset
dataset = glob.glob(f"{path_in}/*cleaned.set")[0]
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

# Get subject id as string
id_string = dataset.split("VP")[1][0:2]


# Load trialinfo
trialinfo = np.genfromtxt(
    dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
)

# Recode distractor and target positions in 2 bins 0-1 (roughly left vs right...)
trialinfo[:, 20] = np.floor((trialinfo[:, 20] - 1) / 4)
trialinfo[:, 21] = np.floor((trialinfo[:, 21] - 1) / 4)

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
# 10: prev_switch
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

# Pick a timepoint
time_idx = 50
X_all = X_list[time_idx]

# Get condition indices
idx_std_rep = (trialinfo[:, 3] == 0) & (trialinfo[:, 9] == 0)
idx_std_swi = (trialinfo[:, 3] == 0) & (trialinfo[:, 9] == 1)
idx_bon_rep = (trialinfo[:, 3] == 1) & (trialinfo[:, 9] == 0)
idx_bon_swi = (trialinfo[:, 3] == 1) & (trialinfo[:, 9] == 1)

# Define classification
clf = {
        "label": "task in std-rep",
        "trial_idx": idx_std_rep,
        "y_col": 4,
    }

# Select features and labels
X = X_all[clf["trial_idx"], :]
y = trialinfo[clf["trial_idx"], clf["y_col"]]

# Do some binning on test and training data
binsize = 5
n_bins = 5000
X0 = do_some_binning(X[y == 0, :], binsize, n_bins)
X1 = do_some_binning(X[y == 1, :], binsize, n_bins)
        
# Stack averages
X = np.vstack((X0, X1))

# New labels
y = np.repeat([0, 1], n_bins)

# Shuffle X and y
X, y = sklearn.utils.shuffle(X, y)    

# Split  data into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2
)

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
output_layer = keras.layers.Dense(2, activation="softmax", name="output")(
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
earlystop_cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=10
)

# Fit the model
history = model.fit(
    X_train,
    y_train,
    epochs=1,
    validation_data=([X_valid], [y_valid]),
    callbacks=[checkpoint_cb, earlystop_cb],
    batch_size=128,
)

# Load best model
model = tf.keras.models.load_model(f"{path_out}callback_time_{time_idx}.h5")

# Evaluate model on test data
acc_classy = acc_classy + model.evaluate(X_test, y_test, verbose=0)













