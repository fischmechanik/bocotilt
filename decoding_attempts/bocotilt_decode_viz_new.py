#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:56:12 2023

@author: plkn
"""

# Imports
import glob
import os
import joblib
import numpy as np
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

smoothing_length = 2

# For smoothing...
def moving_average(x, w=smoothing_length):
    return np.convolve(x, np.ones(w), "valid") / w


# Path vars
path_in = "/mnt/data_dump/bocotilt/3_decoding_data/features_reduced_logreg_smoother_swirep_seperated/"

# Define labels to load
labels = [
    "bonus_vs_standard_in_repeat",
    "bonus_vs_standard_in_switch",
    "task_in_repeat_in_standard",
    "task_in_repeat_in_bonus",
    "task_in_switch_in_standard",
    "task_in_switch_in_bonus",
    "cue_in_repeat_in_standard_in_color",
    "cue_in_repeat_in_bonus_in_color",
    "cue_in_switch_in_standard_in_color",
    "cue_in_switch_in_bonus_in_color",
    "cue_in_repeat_in_standard_in_tilt",
    "cue_in_repeat_in_bonus_in_tilt",
    "cue_in_switch_in_standard_in_tilt",
    "cue_in_switch_in_bonus_in_tilt",
    "response_in_repeat_in_standard_in_color",
    "response_in_repeat_in_bonus_in_color",
    "response_in_switch_in_standard_in_color",
    "response_in_switch_in_bonus_in_color",
    "response_in_repeat_in_standard_in_tilt",
    "response_in_repeat_in_bonus_in_tilt",
    "response_in_switch_in_standard_in_tilt",
    "response_in_switch_in_bonus_in_tilt",
    "target_in_repeat_in_standard_in_color",
    "target_in_repeat_in_bonus_in_color",
    "target_in_switch_in_standard_in_color",
    "target_in_switch_in_bonus_in_color",
    "target_in_repeat_in_standard_in_tilt",
    "target_in_repeat_in_bonus_in_tilt",
    "target_in_switch_in_standard_in_tilt",
    "target_in_switch_in_bonus_in_tilt",
    "distractor_in_repeat_in_standard_in_color",
    "distractor_in_repeat_in_bonus_in_color",
    "distractor_in_switch_in_standard_in_color",
    "distractor_in_switch_in_bonus_in_color",
    "distractor_in_repeat_in_standard_in_tilt",
    "distractor_in_repeat_in_bonus_in_tilt",
    "distractor_in_switch_in_standard_in_tilt",
    "distractor_in_switch_in_bonus_in_tilt",
]

# Result collectors
accs = {}
accs_butterfly = {}

# Loop labels
for label in labels:

    # Get label datasets
    datasets = glob.glob(f"{path_in}/{label}*.joblib")

    # List for decoding task data
    acc = []
    acc_buttrefly = []

    # Loop datasets
    for dataset in datasets:

        # Load datasets
        data = joblib.load(dataset)

        # Get dim vectors
        times = data["times"]
        chan_info = data["info_object"]

        # Adjust times
        times = times[: -(smoothing_length - 1)]

        # Get dims
        n_times, n_chans = (
            len(data["times"]),
            len(data["info_object"].ch_names),
        )

        # Smooth data
        acc_smoothed = moving_average(data["acc"])

        # Collect data
        acc.append(acc_smoothed)
        acc_buttrefly.append(acc_smoothed)

    # Stack data
    acc = np.stack(acc)

    # Save acc data for butterfly
    accs_butterfly[label] = np.stack(acc)

    # Average data
    accs[label] = np.stack(acc).mean(axis=0)


# Define adjacency matrix
adjacency, channel_names = mne.channels.find_ch_adjacency(
    data["info_object"], ch_type="eeg"
)


# Plot bonus decoding
fig = plt.figure()
plt.plot(times, accs["bonus_vs_standard_in_repeat"], label="repeat")
plt.plot(times, accs["bonus_vs_standard_in_switch"], label="switch")
plt.legend()
plt.title("bonus decoding")
fig.show()

# Task decoding =====================================================================================================================

# Get data
rep_std = accs_butterfly["task_in_repeat_in_standard"]
rep_bon = accs_butterfly["task_in_repeat_in_bonus"]
swi_std = accs_butterfly["task_in_switch_in_standard"]
swi_bon = accs_butterfly["task_in_switch_in_bonus"]

# Plot data
fig = plt.figure(figsize=(8, 4))
ax = plt.subplot()
ax.plot(times, rep_std.mean(axis=0), "-c", label="rep-std")
ax.plot(times, rep_bon.mean(axis=0), "-m", label="rep-bon")
ax.plot(times, swi_std.mean(axis=0), ":c", label="swi-std")
ax.plot(times, swi_bon.mean(axis=0), ":m", label="swi-bon")
ax.legend()
ax.set_title("task decoding")

# Combine for tests
tests = []
tests.append(
    {
        "label": "repeat vs switch",
        "data1": (rep_std + rep_bon) / 2,
        "data2": (swi_std + swi_bon) / 2,
    }
)
tests.append(
    {
        "label": "standard vs bonus",
        "data1": (rep_std + swi_std) / 2,
        "data2": (rep_bon + swi_bon) / 2,
    }
)
tests.append(
    {
        "label": "interaction",
        "data1": rep_std * swi_std,
        "data2": rep_bon * swi_bon,
    }
)

# Iterate tests
for test in tests:

    # Perform test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        [test["data1"], test["data2"]],
        n_permutations=5000,
        threshold=3.0,
        tail=0,
        n_jobs=None,
        out_type="mask",
    )

    # Get significant clusters
    test["n_clusters"] = len(clusters)
    for cluster_idx, cluster in enumerate(clusters):
        print(cluster_p_values[cluster_idx])
        cluster = cluster[0]
        if cluster_p_values[cluster_idx] <= 0.1:
            h = ax.axvspan(
                times[cluster.start], times[cluster.stop - 1], color="r", alpha=0.3
            )

# Show figure
fig.show()


# Plot cue decoding
fig = plt.figure()
pd = (
    accs["cue_in_repeat_in_standard_in_color"]
    + accs["cue_in_repeat_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-std")
pd = (
    accs["cue_in_repeat_in_bonus_in_color"] + accs["cue_in_repeat_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-bon")
pd = (
    accs["cue_in_switch_in_standard_in_color"]
    + accs["cue_in_switch_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-std")
pd = (
    accs["cue_in_switch_in_bonus_in_color"] + accs["cue_in_switch_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-bon")
plt.legend()
plt.title("cue decoding")
fig.show()


# Plot response decoding
fig = plt.figure()
pd = (
    accs["response_in_repeat_in_standard_in_color"]
    + accs["response_in_repeat_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-std")
pd = (
    accs["response_in_repeat_in_bonus_in_color"]
    + accs["response_in_repeat_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-bon")
pd = (
    accs["response_in_switch_in_standard_in_color"]
    + accs["response_in_switch_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-std")
pd = (
    accs["response_in_switch_in_bonus_in_color"]
    + accs["response_in_switch_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-bon")
plt.legend()
plt.title("response decoding")
fig.show()


# Plot target decoding
fig = plt.figure()
pd = (
    accs["target_in_repeat_in_standard_in_color"]
    + accs["target_in_repeat_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-std")
pd = (
    accs["target_in_repeat_in_bonus_in_color"]
    + accs["target_in_repeat_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-bon")
pd = (
    accs["target_in_switch_in_standard_in_color"]
    + accs["target_in_switch_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-std")
pd = (
    accs["target_in_switch_in_bonus_in_color"]
    + accs["target_in_switch_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-bon")
plt.legend()
plt.title("target decoding")
fig.show()


# Plot distractor decoding
fig = plt.figure()
pd = (
    accs["distractor_in_repeat_in_standard_in_color"]
    + accs["distractor_in_repeat_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-std")
pd = (
    accs["distractor_in_repeat_in_bonus_in_color"]
    + accs["distractor_in_repeat_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="rep-bon")
pd = (
    accs["distractor_in_switch_in_standard_in_color"]
    + accs["distractor_in_switch_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-std")
pd = (
    accs["distractor_in_switch_in_bonus_in_color"]
    + accs["distractor_in_switch_in_bonus_in_tilt"]
) / 2
plt.plot(times, pd, label="swi-bon")
plt.legend()
plt.title("distractor decoding")
fig.show()
