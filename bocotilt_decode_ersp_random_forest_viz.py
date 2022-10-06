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
path_in = "/mnt/data_dump/bocotilt/3_decoded/"

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
    #ax2.legend((h,), ("cluster p-value < 0.05",))
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("f-values")

    # Tight layout
    fig.tight_layout()


# Average across subjects
ave_bonus = []
ave_task_std = []
ave_task_bon = []
ave_cue_std = []
ave_cue_bon = []
ave_target_std = []
ave_target_bon = []
ave_dist_std = []
ave_dist_bon = []
ave_resp_std = []
ave_resp_bon = []

# read data
for dataset in datasets:

    # Load dataset
    data = joblib.load(dataset)

    # Task decoding
    ave_task_std.append(moving_average(data["acc"][0]))
    ave_task_bon.append(moving_average(data["acc"][1]))

    # Bonus decoding
    tmp1 = (data["acc"][2] + data["acc"][3]) / 2
    ave_bonus.append(moving_average(tmp1))

    # Cue decoding
    tmp1 = (data["acc"][4] + data["acc"][5]) / 2
    tmp2 = (data["acc"][6] + data["acc"][7]) / 2
    ave_cue_std.append(moving_average(tmp1))
    ave_cue_bon.append(moving_average(tmp2))

    # Response decoding
    tmp1 = (data["acc"][8] + data["acc"][9]) / 2
    tmp2 = (data["acc"][10] + data["acc"][11]) / 2
    ave_resp_std.append(moving_average(tmp1))
    ave_resp_bon.append(moving_average(tmp2))

    # Target decoding
    tmp1 = (data["acc"][12] + data["acc"][13]) / 2
    tmp2 = (data["acc"][14] + data["acc"][15]) / 2
    ave_target_std.append(moving_average(tmp1))
    ave_target_bon.append(moving_average(tmp2))

    # Distractor decoding
    tmp1 = (data["acc"][16] + data["acc"][17]) / 2
    tmp2 = (data["acc"][18] + data["acc"][19]) / 2
    ave_dist_std.append(moving_average(tmp1))
    ave_dist_bon.append(moving_average(tmp2))

# Adjust time vector to smoothing function
times = data["tf_times"][smowin - 1 :]

# get condition data
condition1 = np.stack(ave_task_std)
condition2 = np.stack(ave_task_bon)

# Task
plot_decoding_result(
    condition1,
    condition2,
    decode_label="task",
    label_cond_1="standard",
    label_cond_2="bonus",
    f_thresh=2.0,
)

# get condition data
condition1 = np.stack(ave_cue_std)
condition2 = np.stack(ave_cue_bon)

# Cue 
plot_decoding_result(
    condition1,
    condition2,
    decode_label="cue",
    label_cond_1="standard",
    label_cond_2="bonus",
    f_thresh=2.0,
)

# get condition data
condition1 = np.stack(ave_target_std)
condition2 = np.stack(ave_target_bon)

# Target position
plot_decoding_result(
    condition1,
    condition2,
    decode_label="target position",
    label_cond_1="standard",
    label_cond_2="bonus",
    f_thresh=2.0,
)

