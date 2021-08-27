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

# Path decoding data
path_in = "/mnt/data2/bocotilt/3_decoded/"

# Get list of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Load dataset
data = joblib.load(datasets[0])

clf_labels = data["decode_labels"]
times = data["eeg_times"]

# Bonus versus standard trials
plt.subplot(111)
plt.title("bonus versus standard")
plt.plot(times, data["acc_true"][0], label="observed", color="c")
plt.plot(times, data["acc_fake"][0], label="chance", color="m")
plt.ylabel("accuracy")
plt.legend()

# Task decoding
plt.subplot(111)
plt.title("task decoding")
plt.plot(times, data["acc_true"][1], label="standard", color="c")
plt.plot(times, data["acc_true"][2], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Task switch decoding
plt.subplot(111)
plt.title("task switch decoding")
plt.plot(times, data["acc_true"][17], label="standard", color="c")
plt.plot(times, data["acc_true"][18], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Response interference decoding
plt.subplot(111)
plt.title("response interference decoding")
plt.plot(times, data["acc_true"][15], label="standard", color="c")
plt.plot(times, data["acc_true"][16], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Cue decoding
plt.subplot(111)
plt.title("cue decoding")
plt.plot(times, data["acc_true"][3], label="color std", color="c")
plt.plot(times, data["acc_true"][4], label="tilt std", color="m")
plt.plot(times, data["acc_true"][5], label="color bonus", color="g")
plt.plot(times, data["acc_true"][6], label="tilt bonus", color="r")
plt.ylabel("accuracy")
plt.legend()

# Target decoding
plt.subplot(111)
plt.title("target decoding")
plt.plot(times, data["acc_true"][7], label="color std", color="c")
plt.plot(times, data["acc_true"][8], label="tilt std", color="m")
plt.plot(times, data["acc_true"][9], label="color bonus", color="g")
plt.plot(times, data["acc_true"][10], label="tilt bonus", color="r")
plt.ylabel("accuracy")
plt.legend()

# Distractor decoding
plt.subplot(111)
plt.title("distractor decoding")
plt.plot(times, data["acc_true"][11], label="color std", color="c")
plt.plot(times, data["acc_true"][12], label="tilt std", color="m")
plt.plot(times, data["acc_true"][13], label="color bonus", color="g")
plt.plot(times, data["acc_true"][14], label="tilt bonus", color="r")
plt.ylabel("accuracy")
plt.legend()
    