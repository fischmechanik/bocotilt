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
    data_cond_1,
    condition2,
    decode_label="title",
    label_cond_1="cond1",
    label_cond_2="cond2",
    performance_measure="accuracy",
    f_thresh=6.0,
):

    # Perform cluster-test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
        [data_cond_1, condition2],
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
        times, data_cond_1.mean(axis=0), label=label_cond_1,
    )
    ax1.plot(
        times, condition2.mean(axis=0), label=label_cond_2,
    )
    ax1.set_ylabel(performance_measure)
    ax1.set_xlabel("time (s)")
    ax1.legend()

    # Plot statistics
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = ax2.axvspan(times[c.start], times[c.stop - 1], color="g", alpha=0.3)
        else:
            ax2.axvspan(
                times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3
            )

    hf = plt.plot(times, T_obs, "m")
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
    condition1,
    condition2,
    decode_label="stuff",
    label_cond_1="lab1",
    label_cond_2="lab2",
    f_thresh=2.0,
)

