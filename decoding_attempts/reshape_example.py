#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:34:34 2023

@author: plkn
"""

# Imports
import numpy as np

# trial x chan x freq
d3d = np.array([
    [
        [
            "t1c1f1",
            "t1c1f2",
            "t1c1f3",
            "t1c1f4",
        ],
        [
            "t1c2f1",
            "t1c2f2",
            "t1c2f3",
            "t1c2f4",
        ],
        [
            "t1c3f1",
            "t1c3f2",
            "t1c3f3",
            "t1c3f4",
        ],
    ],
    [
        [
            "t2c1f1",
            "t2c1f2",
            "t2c1f3",
            "t2c1f4",
        ],
        [
            "t2c2f1",
            "t2c2f2",
            "t2c2f3",
            "t2c2f4",
        ],
        [
            "t2c3f1",
            "t2c3f2",
            "t2c3f3",
            "t2c3f4",
        ],
    ],
])

# Get dims
n_trials, n_channels, n_freqs = d3d.shape

# to 2d
d2d = d3d.reshape((n_trials, n_channels * n_freqs))

# Back
d3d_after = d2d.reshape((n_trials, n_channels, n_freqs))

















