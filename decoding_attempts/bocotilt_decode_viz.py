#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:56:12 2023

@author: plkn
"""

# Imports
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
import random
import matplotlib.pyplot as plt

# Path vars
path_in = "/mnt/data_dump/bocotilt/3_decoding_data/"

# Define labels to load
labels = [
    "bonus_vs_standard",
    "task_in_bonus",
    "task_in_standard",
]

# Result collectors
accs = {}
fmps = {}

# Loop labels
for label in labels:

    # Get label datasets
    datasets = glob.glob(f"{path_in}/{label}*09.joblib")

    # List for decoding task data
    acc = []
    fmp = []

    # Loop datasets
    for dataset in datasets:

        # Load datasets
        data = joblib.load(dataset)

        # Get dim vectors
        times = data["times"]
        freqs = data["freqs"]
        chan_info = data["info_object"]

        # Get dims
        n_times, n_freqs, n_chans = (
            len(data["times"]),
            len(data["freqs"]),
            len(data["info_object"].ch_names),
        )

        # Collect data
        acc.append(data["acc"])
        fmp.append(data["fmp"])

    # Stack data
    acc = np.stack(acc)
    fmp = np.stack(fmp)

    # Average data
    accs[label] = np.stack(acc).mean(axis=0)
    fmps[label] = np.stack(fmp).mean(axis=0)

# Plot bonus decoding
fig = plt.figure()
plt.plot(times, accs["bonus_vs_standard"], label="bon")
plt.legend()
plt.title("bonus decoding")
fig.show()

# Plot task decoding
fig = plt.figure()
plt.plot(times, accs["task_in_bonus"], label="bon")
plt.plot(times, accs["task_in_standard"], label="std")
plt.legend()
plt.title("task decoding")
fig.show()

# Plot fmp
chan_nums = np.arange(1, 128)
time_idx = (times >= 0.3) & (times <= 0.4)
pd = fmps["task_in_standard"][time_idx, :, :].mean(axis=0).T
fig = plt.figure()
plt.contourf(chan_nums, freqs, pd, cmap="plasma")
plt.clim((0, 0.004))
plt.colorbar()

# Plot fmp
pd = fmps["bonus_vs_standard"][:, :, :].mean(axis=2).T
fig = plt.figure()
plt.contourf(times, chan_nums, pd, cmap="plasma")
plt.clim((0, 0.002))
plt.colorbar()

# Plot fmp
pd = fmps["task_in_standard"][:, :, :].mean(axis=1).T
fig = plt.figure()
plt.contourf(times, freqs, pd, cmap="plasma")
plt.clim((0, 0.004))
plt.colorbar()
