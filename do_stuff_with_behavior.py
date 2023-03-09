#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:40:48 2023

@author: plkn
"""
# Imports
import numpy as np
import os

# Paths
path_in = "/home/plkn/bocotilt/behavior/"

# Load data
data = np.genfromtxt(os.path.join(path_in, "behavioral_data.csv"), delimiter=",")