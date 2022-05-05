#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from cool_colormaps import cga_p1_dark as ccm

# Define paths
path_in = "/mnt/data_dump/bocotilt/6_tf_analysis/"

# Load data
fn = os.path.join(path_in, "tf_datasets_block_bonus.joblib")
tf_datasets = joblib.load(fn)

# Get info
factors = tf_datasets[0]["factors"]
times = tf_datasets[0]["power"][0].times
freqs = tf_datasets[0]["power"][0].freqs

# Get grand averages
grand_averages = []
for condition_idx in range(len(tf_datasets[0]["levels"])):

    # Get factor levels
    condition_levels = tf_datasets[0]["levels"][condition_idx]

    # Create label
    condition_label = "_".join(
        [f"{f}_{condition_levels[f_idx]}" for f_idx, f in enumerate(factors)]
    )

    # Calculate ga
    grand_averages.append(
        {
            "label": condition_label,
            "power": mne.grand_average(
                [tfd["power"][condition_idx] for tfd in tf_datasets]
            ),
            "itc": mne.grand_average(
                [tfd["itc"][condition_idx] for tfd in tf_datasets]
            ),
        }
    )




# Plot 
channel_idx = 126
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(6, 2)
fig.suptitle('Sharing x per column, y per row')
plot_data = grand_averages[0]["power"].data[channel_idx, :, :]
ax1.contourf(times, freqs, plot_data, levels=50, vmin=-0.3, vmax=0.3)
plot_data = grand_averages[0]["power"].data[channel_idx, :, :]
ax2.contourf(times, freqs, plot_data, levels=50, vmin=-0.3, vmax=0.3)
plot_data = grand_averages[0]["power"].data[channel_idx, :, :]
ax3.contourf(times, freqs, plot_data, levels=50, vmin=-0.3, vmax=0.3)
plot_data = grand_averages[0]["power"].data[channel_idx, :, :]
ax4.contourf(times, freqs, plot_data, levels=50, vmin=-0.3, vmax=0.3)



for ax in fig.get_axes():
    ax.label_outer()
    
    
# Grand averages
# ga = mne.grand_average(power_list)


# Plotty
# ga.plot_joint(
#    tmin=-0.5, tmax=2, timefreqs=[(0.5, 10), (1.1, 4)], cmap="PuOr", vmin=-0.3, vmax=0.3
# )
