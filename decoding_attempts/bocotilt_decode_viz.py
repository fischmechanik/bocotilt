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

smoothing_length = 5

# For smoothing...
def moving_average(x, w=smoothing_length):
    return np.convolve(x, np.ones(w), 'valid') / w

# Path vars
path_in = "/mnt/data_dump/bocotilt/3_decoding_data/features_reduced/"

# Define labels to load
labels = [
    "bonus_vs_standard",
    "task_in_bonus",
    "task_in_standard",
    "cue_in_standard_in_color",
    "cue_in_standard_in_tilt",
    "cue_in_bonus_in_color",
    "cue_in_bonus_in_tilt",
    "response_in_standard_in_color",
    "response_in_standard_in_tilt",
    "response_in_bonus_in_color",
    "response_in_bonus_in_tilt",
    "target_in_standard_in_color",
    "target_in_standard_in_tilt",
    "target_in_bonus_in_color",
    "target_in_bonus_in_tilt",
    "distractor_in_standard_in_color",
    "distractor_in_standard_in_tilt",
    "distractor_in_bonus_in_color",
    "distractor_in_bonus_in_tilt",
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
        times = times[:-(smoothing_length-1)]

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
plt.plot(times, accs["bonus_vs_standard"], label="bon")
plt.legend()
plt.title("bonus decoding")
fig.show()

data_1 = accs_butterfly["bonus_vs_standard"] 
data_2 = np.zeros(accs_butterfly["bonus_vs_standard"].shape) + 0.5

threshold = 1.0
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([data_1, data_2], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=None,
                             out_type='mask')


fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title('title')
ax.plot(times, accs["task_in_standard"] - accs["task_in_bonus"],
        label="task delta")
ax.set_ylabel("acc")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax2.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)

hf = plt.plot(times, T_obs, 'g')
ax2.legend((h, ), ('cluster p-value < 0.05', ))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")





# Plot task decoding
fig = plt.figure()
plt.plot(times, accs["task_in_bonus"], label="bon")
plt.plot(times, accs["task_in_standard"], label="std")
plt.legend()
plt.title("task decoding")
fig.show()



data_1 = accs_butterfly["task_in_bonus"] 
data_2 = accs_butterfly["task_in_standard"] 

threshold = 1.0
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([data_1, data_2], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=None,
                             out_type='mask')


fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title('title')
ax.plot(times, accs["task_in_standard"] - accs["task_in_bonus"],
        label="task delta")
ax.set_ylabel("acc")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax2.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)

hf = plt.plot(times, T_obs, 'g')
ax2.legend((h, ), ('cluster p-value < 0.05', ))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")








# Plot cue decoding
fig = plt.figure()
pd = (accs["cue_in_standard_in_color"] + accs["cue_in_standard_in_tilt"]) / 2
plt.plot(times, pd, label="std")
pd = (accs["cue_in_bonus_in_color"] + accs["cue_in_bonus_in_tilt"]) / 2
plt.plot(times, pd, label="bon")
plt.legend()
plt.title("cue decoding")
fig.show()

# Plot response decoding
fig = plt.figure()
pd = (accs["response_in_standard_in_color"] + accs["response_in_standard_in_tilt"]) / 2
plt.plot(times, pd, label="std")
pd = (accs["response_in_bonus_in_color"] + accs["response_in_bonus_in_tilt"]) / 2
plt.plot(times, pd, label="bon")
plt.legend()
plt.title("response decoding")
fig.show()

# Plot target decoding
fig = plt.figure()
pd = (accs["target_in_standard_in_color"] + accs["target_in_standard_in_tilt"]) / 2
plt.plot(times, pd, label="std")
pd = (accs["target_in_bonus_in_color"] + accs["target_in_bonus_in_tilt"]) / 2
plt.plot(times, pd, label="bon")
plt.legend()
plt.title("target decoding")
fig.show()

# Plot distractor decoding
fig = plt.figure()
pd = (
    accs["distractor_in_standard_in_color"] + accs["distractor_in_standard_in_tilt"]
) / 2
plt.plot(times, pd, label="std")
pd = (accs["distractor_in_bonus_in_color"] + accs["distractor_in_bonus_in_tilt"]) / 2
plt.plot(times, pd, label="bon")
plt.legend()
plt.title("distractor decoding")
fig.show()











