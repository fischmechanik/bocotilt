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
path_in = "/mnt/data_dump/bocotilt/3_decoded/"

# Get list of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Load dataset
data = joblib.load(datasets[1])

clf_labels = data["decode_labels"]
times = data["tf_times"]

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

# Cue decoding
plt.subplot(111)
plt.title("cue decoding")
plt.plot(times, data["acc_true"][3], label="standard", color="c")
plt.plot(times, data["acc_true"][4], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Target decoding
plt.subplot(111)
plt.title("target decoding")
plt.plot(times, data["acc_true"][5], label="standard", color="c")
plt.plot(times, data["acc_true"][6], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Distractor decoding
plt.subplot(111)
plt.title("distractor decoding")
plt.plot(times, data["acc_true"][7], label="standard", color="c")
plt.plot(times, data["acc_true"][8], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()

# Response decoding
plt.subplot(111)
plt.title("response decoding")
plt.plot(times, data["acc_true"][9], label="standard", color="c")
plt.plot(times, data["acc_true"][10], label="bonus", color="m")
plt.ylabel("accuracy")
plt.legend()   