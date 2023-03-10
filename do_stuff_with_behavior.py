#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:40:48 2023

@author: plkn
"""
# Imports
import numpy as np
import os
import pyddm as ddm
import pandas as pd

# Paths
path_in = "/home/plkn/bocotilt/behavior/"

# Load data
data = np.genfromtxt(os.path.join(path_in, "behavioral_data.csv"), delimiter=",")

# Variable names
varnames = [
    "id",
    "block_nr",
    "trial_nr",
    "bonustrial",
    "tilt_task",
    "cue_ax",
    "target_red_left",
    "distractor_red_left",
    "response_interference",
    "task_switch",
    "prev_switch",
    "prev_accuracy",
    "correct_response",
    "response_side",
    "rt",
    "rt_thresh_color",
    "rt_thresh_tilt",
    "accuracy",
    "position_color",
    "position_tilt",
    "position_target",
    "position_distractor",
    "sequence_pos",
]

# Create data frame
df = pd.DataFrame(data=data, columns=varnames)

# Select subject
df_id = df[df["id"] == 9]

# Exclude outlier RTs
df_id = df_id[df_id["rt"] > 100]
df_id = df_id[df_id["rt"] < 1200]


# Create a sample object from our data.  This is the standard input
# format for fitting procedures.  Since RT and correct/error are
# both mandatory columns, their names are specified by command line
# arguments.
a_sample = ddm.Sample.from_pandas_dataframe(
    df_id, rt_column_name="rt", choice_column_name="correct"
)


class DriftCoherence(ddm.models.Drift):
    name = "Drift depends linearly on coherence"
    required_parameters = ["driftcoh"]  # <-- Parameters we want to include in the model
    required_conditions = [
        "coh"
    ]  # <-- Task parameters ("conditions"). Should be the same name as in the sample.

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.driftcoh * conditions["coh"]
