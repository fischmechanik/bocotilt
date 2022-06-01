#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:35:36 2021

@author: plkn
"""

# Import
import joblib
import glob
import matplotlib.pyplot as plt
import numpy as np
import mne

# Path decoding data
path_in = "/mnt/data2/bocotilt/3_decoded_logistic_regression/"

# Get list of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Smoothin factor
smowin = 5

# A smoothening function
def moving_average(x, w=smowin):
    return np.convolve(x, np.ones(w), "valid") / w


# A plotting and statistics function
def plot_decoding_result(
    data_std_rep,
    data_std_swi,
    data_bon_rep,
    data_bon_swi,
    decode_label="title",
    f_thresh=6.0,
):

    # Average for main effects
    data_std = np.stack([(data_std_rep[x] + data_std_swi[x]) / 2 for x in range(len(data_std_rep))])
    data_bon = np.stack([(data_bon_rep[x] + data_bon_swi[x]) / 2 for x in range(len(data_bon_rep))])
    data_rep = np.stack([(data_std_rep[x] + data_bon_rep[x]) / 2 for x in range(len(data_std_rep))])
    data_swi = np.stack([(data_std_swi[x] + data_bon_swi[x]) / 2 for x in range(len(data_std_swi))])
    
    # Stack
    data_std_rep = np.stack(data_std_rep)
    data_std_swi = np.stack(data_std_swi)
    data_bon_rep = np.stack(data_bon_rep)
    data_bon_swi = np.stack(data_bon_swi)
    
    # Test bonus
    T_obs_bon, clusters_bon, cluster_p_values_bon, H0_bon = mne.stats.permutation_cluster_test(
        [data_std, data_bon],
        n_permutations=1000,
        threshold=f_thresh,
        tail=1,
        n_jobs=1,
        out_type="mask",
    )
    
    # Test switch
    T_obs_swi, clusters_swi, cluster_p_values_swi, H0_swi = mne.stats.permutation_cluster_test(
        [data_rep, data_swi],
        n_permutations=1000,
        threshold=f_thresh,
        tail=1,
        n_jobs=1,
        out_type="mask",
    )

    # Create 2-axis figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

    # Set figure title
    fig.suptitle(decode_label, fontsize=12)

    # Plot classifier performance
    ax1.plot(
        times, data_std_rep.mean(axis=0), label="std-rep",
    )
    ax1.plot(
        times, data_std_swi.mean(axis=0), label="std-swi",
    )
    ax1.plot(
        times, data_bon_rep.mean(axis=0), label="bon-rep",
    )
    ax1.plot(
        times, data_bon_swi.mean(axis=0), label="bon-swi",
    )
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("time (s)")
    ax1.legend()

    # Plot statistics
    #for i_c, c in enumerate(clusters_bon):
    #    c = c[0]
    #    if cluster_p_values[i_c] <= 0.05:
    #        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="g", alpha=0.3)
    #    else:
    #        ax2.axvspan(
    #            times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3
    #        )

    #hf = plt.plot(times, T_obs_bon, "m")
    # ax2.legend((h,), ("cluster p-value < 0.05",))
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("f-values")

    # Tight layout
    fig.tight_layout()


# Average across subjects
task_std_rep = []
task_std_swi = []
task_bon_rep = []
task_bon_swi = []
cue_std_rep = []
cue_std_swi = []
cue_bon_rep = []
cue_bon_swi = []

# read data
for dataset in datasets:

    # Load dataset
    data = joblib.load(dataset)

    # Task decoding
    task_std_rep.append(moving_average(data["acc"][0]))
    task_std_swi.append(moving_average(data["acc"][1]))
    task_bon_rep.append(moving_average(data["acc"][2]))
    task_bon_swi.append(moving_average(data["acc"][3]))

    cue_std_rep.append(moving_average(data["acc"][8]))
    cue_std_swi.append(moving_average(data["acc"][9]))
    cue_bon_rep.append(moving_average(data["acc"][10]))
    cue_bon_swi.append(moving_average(data["acc"][11]))


# Adjust time vector to smoothing function
times = data["tf_times"][smowin - 1 :]

# get condition data
condition1 = np.stack(cue_std_rep)
condition2 = np.stack(cue_bon_rep)

# Target position
plot_decoding_result(
    task_std_rep,
    task_std_swi,
    task_bon_rep,
    task_bon_swi,
    decode_label="stuff",
    f_thresh=2.0,
)

