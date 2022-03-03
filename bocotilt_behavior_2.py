# Imports
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.anova

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

# Eclude the strange pos 9 trial
data = np.delete(data, data[:, 21] == 9, axis=0)

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

# Add binarized sequential positions
binarized_seqpos = np.zeros((data.shape[0], 1)) - 1
binarized_seqpos[data[:, 21] <= 3, 0] = 0
binarized_seqpos[data[:, 21] >= 7, 0] = 1
data = np.hstack((data, binarized_seqpos))

# Remove non defined
data = data[data[:, 16] != 2, :]
#data = data[data[:, 9] != -1, :]
data = data[data[:, 1] > 4, :]
data = data[data[:, 23] >= 0, :]

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
    "seqpos_binarized",
]

# Create df
df = pd.DataFrame(data=data, columns=columns)

# Get unit-condition averages
df2 = df.groupby(["id", "bonustrial", "seqpos_binarized"], as_index=False)[
    "log_rt", "log_accuracy"
].mean()

# Draw rt averages
g = sns.catplot(
    x="bonustrial",
    y="log_rt",
    hue="seqpos_binarized",
    #col = "task_switch",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df2,
)
g.despine(left=True)

# Draw accuracy averages
g = sns.catplot(
    x="bonustrial",
    y="log_accuracy",
    hue="seqpos_binarized",
    #col = "task_switch",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df2,
)
g.despine(left=True)

# DataFrame for statistics
df3 = df.groupby(
    ["id", "bonustrial",  "seqpos_binarized"], as_index=False
)["log_rt", "log_accuracy"].mean()

# Calculate anova rt
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df3,
    depvar="log_rt",
    subject="id",
    within=["bonustrial", "seqpos_binarized"],
).fit()
print(anova_out)

# Calculate anova accuracy
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df3,
    depvar="log_accuracy",
    subject="id",
    within=["bonustrial", "seqpos_binarized"],
).fit()
print(anova_out)

