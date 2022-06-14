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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define paths
path_in = "/mnt/data_dump/bocotilt/4_ersp/"
path_clean_data = "/mnt/data_dump/bocotilt/2_autocleaned/"

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

# Define adjacencies for tf-space, and tf only
tfs_adjacency = mne.stats.combine_adjacency(len(freqs), len(times), adjacency)
tf_adjacency = mne.stats.combine_adjacency(len(freqs), len(times))

# Get data as matrices (subject x freqs x times x channels)
pow_std_rep = np.stack(
    [tfd["power"][0].data.transpose((1, 2, 0)) for tfd in tf_datasets]
)
pow_std_swi = np.stack(
    [tfd["power"][1].data.transpose((1, 2, 0)) for tfd in tf_datasets]
)
pow_bon_rep = np.stack(
    [tfd["power"][2].data.transpose((1, 2, 0)) for tfd in tf_datasets]
)
pow_bon_swi = np.stack(
    [tfd["power"][3].data.transpose((1, 2, 0)) for tfd in tf_datasets]
)

# Cluster test
data1 = pow_std_rep - pow_std_swi
data2 = pow_bon_rep - pow_bon_swi

#data1 = (pow_std_rep + pow_std_swi) / 2
#data2 = (pow_bon_rep + pow_bon_swi) / 2

#data1 = (pow_std_rep + pow_bon_rep) / 2
#data2 = (pow_std_swi + pow_bon_swi) / 2

F_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_test(
    [data1, data2],
    threshold=None,
    n_permutations=256,
    tail=0,
    stat_fun=None,
    adjacency=tfs_adjacency,
    n_jobs=-2,
    seed=42,
    max_step=1,
    exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
)

good_cluster_inds = np.where(cluster_pv < 0.7)[0]

for i_clu, clu_idx in enumerate(good_cluster_inds):
    
    # unpack cluster information, get unique indices
    freq_inds, time_inds, space_inds = cluster[clu_idx]
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    freq_inds = np.unique(freq_inds)

    # get topography for F stat
    f_map = F_obs[freq_inds].mean(axis=0)
    f_map = f_map[time_inds].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = times[time_inds]

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], dummy_epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vmin=np.min, vmax=np.max, show=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # add new axis for spectrogram
    ax_spec = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += " (max over channels)"
    F_obs_plot = F_obs[..., ch_inds].max(axis=-1)
    F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
    F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
        F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

    for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
        c = ax_spec.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                           extent=[times[0], times[-1],
                                   freqs[0], freqs[-1]])
    ax_spec.set_xlabel('Time (ms)')
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title(title)

    # add another colorbar
    ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(c, cax=ax_colorbar2)
    ax_colorbar2.set_ylabel('F-stat')

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
