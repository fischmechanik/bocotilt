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

# Path decoding data
path_in = "/mnt/data_dump/bocotilt/3_decoded/"

# Get list of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Smoothin factor
smowin = 5

# A smoothening function
def moving_average(x, w=smowin):
    return np.convolve(x, np.ones(w), "valid") / w


# Average across subjects
ave_bonus_true = []
ave_bonus_fake = []
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

for dataset in datasets:

    # Load dataset
    data = joblib.load(dataset)
    
    # Task decoding
    ave_task_std.append(moving_average(data["acc_true"][0]))
    ave_task_bon.append(moving_average(data["acc_true"][1]))

    # Bonus decoding
    tmp1 = (data["acc_true"][2] + data["acc_true"][3]) / 2
    tmp2 = (data["acc_fake"][2] + data["acc_fake"][3]) / 2
    ave_bonus_true.append(moving_average(tmp1))
    ave_bonus_fake.append(moving_average(tmp2))
    
    # Cue decoding
    tmp1 = (data["acc_true"][4] + data["acc_true"][5]) / 2
    tmp2 = (data["acc_true"][6] + data["acc_true"][7]) / 2
    ave_cue_std.append(moving_average(tmp1))
    ave_cue_bon.append(moving_average(tmp2))
    
    # Response decoding
    tmp1 = (data["acc_true"][8] + data["acc_true"][9]) / 2
    tmp2 = (data["acc_true"][10] + data["acc_true"][11]) / 2
    ave_resp_std.append(moving_average(tmp1))
    ave_resp_bon.append(moving_average(tmp2))

    # Target decoding
    tmp1 = (data["acc_true"][12] + data["acc_true"][13]) / 2
    tmp2 = (data["acc_true"][14] + data["acc_true"][15]) / 2
    ave_target_std.append(moving_average(tmp1))
    ave_target_bon.append(moving_average(tmp2))
    
    # Distractor decoding
    tmp1 = (data["acc_true"][16] + data["acc_true"][17]) / 2
    tmp2 = (data["acc_true"][18] + data["acc_true"][19]) / 2
    ave_dist_std.append(moving_average(tmp1))
    ave_dist_bon.append(moving_average(tmp2))

ave_task_std = np.mean(ave_task_std, axis=0)
ave_task_bon = np.mean(ave_task_bon, axis=0)
ave_bonus_true = np.mean(ave_bonus_true, axis=0)
ave_bonus_fake = np.mean(ave_bonus_fake, axis=0)
ave_cue_std = np.mean(ave_cue_std, axis=0)
ave_cue_bon = np.mean(ave_cue_bon, axis=0)
ave_resp_std = np.mean(ave_resp_std, axis=0)
ave_resp_bon = np.mean(ave_resp_bon, axis=0)
ave_target_std = np.mean(ave_target_std, axis=0)
ave_target_bon = np.mean(ave_target_bon, axis=0)
ave_dist_std = np.mean(ave_dist_std, axis=0)
ave_dist_bon = np.mean(ave_dist_bon, axis=0)


# Load dataset
data = joblib.load(datasets[0])

# Crop times
times = data["tf_times"]
pruneframes = int(np.floor(smowin / 2))
times = times[pruneframes:-pruneframes]

# Bonus versus standard trials
plt.subplot(231)
plt.title("bonus versus standard")
plt.plot(times, ave_bonus_true, label="observed", color="m")
plt.plot(times, ave_bonus_fake, label="chance", color="c")
# plt.legend()

# Task decoding
plt.subplot(232)
plt.title("task set")
plt.plot(times, ave_task_std, label="standard", color="c")
plt.plot(times, ave_task_bon, label="bonus", color="m")
# plt.legend()

# Cue decoding
plt.subplot(233)
plt.title("cue (x vs. y)")
plt.plot(times, ave_cue_std, label="standard", color="c")
plt.plot(times, ave_cue_bon, label="bonus", color="m")
# plt.legend()

# Target decoding
plt.subplot(234)
plt.title("target position")
plt.plot(times, ave_target_std, label="standard", color="c")
plt.plot(times, ave_target_bon, label="bonus", color="m")
# plt.legend()

# Distractor decoding
plt.subplot(235)
plt.title("distractor position")
plt.plot(times, ave_dist_std, label="standard", color="c")
plt.plot(times, ave_dist_bon, label="bonus", color="m")
# plt.legend()

# Response decoding
plt.subplot(236)
plt.title("response side")
plt.plot(times, ave_resp_std, label="standard", color="c")
plt.plot(times, ave_resp_bon, label="bonus", color="m")
# plt.legend()

plt.tight_layout()
