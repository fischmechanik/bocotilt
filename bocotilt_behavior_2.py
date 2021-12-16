# Imports
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Read datasets
data = []
for dataset_idx, dataset in enumerate(datasets):
    data.append(np.genfromtxt(dataset, delimiter=","))

# Stack data
data = np.vstack(data)

# Columns of data
#  0: id
#  1: block_nr
#  2: trial_nr
#  3: bonustrial
#  4: tilt_task
#  5: cue_ax
#  6: target_red_left
#  7: distractor_red_left
#  8: response_interference
#  9: task_switch
# 10: correct_response
# 11: response_side
# 12: rt
# 13: accuracy
# 14: log_response_side
# 15: log_rt
# 16: log_accuracy
# 17: position_color
# 18: position_tilt
# 19: position_target
# 20: position_distractor
# 21: sequence_position
# 22: sequence_length

# Set sequence to non-defined if position 1
data[data[:, 21] == 1, 9] = -1

# Remove non defined
data = data[data[:, 16] != 2, :]
data = data[data[:, 9] != -1, :]
data = data[data[:, 1] > 4, :]

# Define columns
columns = [
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
    "correct_response",
    "response_side",
    "rt",
    "accuracy",
    "log_response_side",
    "log_rt",
    "log_accuracy",
    "position_color",
    "position_tilt",
    "position_target",
    "position_distractor",
    "sequence_position",
    "sequence_length",
]

# Create df
df = pd.DataFrame(data=data, columns=columns)


# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    x="sequence_position",
    y="log_rt",
    hue="bonustrial",
    col="task_switch",
    capsize=0.2,
    palette="Dark2",
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
g.despine(left=True)


# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    x="sequence_position",
    y="log_accuracy",
    hue="bonustrial",
    col="task_switch",
    capsize=0.2,
    palette="Dark2",
    height=6,
    aspect=0.75,
    kind="point",
    data=df,
)
g.despine(left=True)
