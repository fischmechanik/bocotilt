#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import pandas as pd
import joblib
import os
import scipy.io
import matplotlib.pyplot as plt
from cool_colormaps import cga_p1_dark as ccm

# Define paths
path_in = "/mnt/data2/bocotilt/4_ersp/"
path_clean_data = "/mnt/data2/bocotilt/2_autocleaned/"

# Load data
fn = os.path.join(path_in, "tf_datasets_task_switch_bonus.joblib")
tf_datasets = joblib.load(fn)

# Get info
conditions = tf_datasets[0]["conditions"]
times = tf_datasets[0]["power"][0].times
freqs = tf_datasets[0]["power"][0].freqs

# Load channel labels
channel_labels = str(
    scipy.io.loadmat(os.path.join(path_clean_data, "channel_labels.mat"))[
        "channel_labels"
    ]
)[3:-2].split(" ")

# Replace O9 and O10 with I1 and I2
channel_labels = ["I1" if ch == "O9" else ch for ch in channel_labels]
channel_labels = ["I2" if ch == "O10" else ch for ch in channel_labels]

# Create a basic mne info structure
sfreq = 100
info = mne.create_info(channel_labels, sfreq, ch_types="eeg", verbose=None)

# Create a montage
standard_montage = mne.channels.make_standard_montage("standard_1005")

# Create mne epochs objects
dummy_epochs = mne.EpochsArray(np.zeros((10, len(channel_labels), len(times))), info)
dummy_epochs.set_montage(standard_montage)

# Define channel adjacency matrix
adjacency, channel_labels = mne.channels.find_ch_adjacency(
    dummy_epochs.info, ch_type=None
)

# Define adjacency in time and frq domain as well...
tf_adjacency = mne.stats.combine_adjacency(len(freqs), len(times), adjacency)

# Get data as matrices (subject x freqs x times x channels)
pow_std_rep = np.stack([tfd["power"][0].data.transpose((1, 2, 0)) for tfd in tf_datasets])
pow_std_swi = np.stack([tfd["power"][1].data.transpose((1, 2, 0)) for tfd in tf_datasets])
pow_bon_rep = np.stack([tfd["power"][2].data.transpose((1, 2, 0)) for tfd in tf_datasets])
pow_bon_swi = np.stack([tfd["power"][3].data.transpose((1, 2, 0)) for tfd in tf_datasets])


# Cluster test
data1 = (pow_std_rep + pow_std_swi) / 2
data2 = (pow_bon_rep + pow_bon_swi) / 2
F_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_test(
    [data1, data2],
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=tf_adjacency,
    n_jobs=-2,
    seed=None,
    max_step=1,
    exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
)


for ax in fig.get_axes():
    ax.label_outer()


# Plot
channel_idx = 126
clim = 0.4
cmap = "jet"
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("Sharing x per column, y per row")
plot_data_1 = grand_averages[0]["power"].data[channel_idx, :, :]
ax1.contourf(times, freqs, plot_data_1, levels=50, vmin=-clim, vmax=clim, cmap=cmap)
plot_data_2 = grand_averages[1]["power"].data[channel_idx, :, :]
ax2.contourf(times, freqs, plot_data_2, levels=50, vmin=-clim, vmax=clim, cmap=cmap)
plot_data_3 = grand_averages[2]["power"].data[channel_idx, :, :]
ax3.contourf(times, freqs, plot_data_3, levels=50, vmin=-clim, vmax=clim, cmap=cmap)
plot_data_4 = grand_averages[3]["power"].data[channel_idx, :, :]
ax4.contourf(times, freqs, plot_data_4, levels=50, vmin=-clim, vmax=clim, cmap=cmap)
