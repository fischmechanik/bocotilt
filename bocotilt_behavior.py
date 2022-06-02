# Imports
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io
import statsmodels.stats.anova
import statsmodels.formula.api as smf
import os

# Paths
path_in = "/mnt/data2/bocotilt/2_autocleaned/"
path_veusz = "/mnt/data2/bocotilt/4_ersp/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Read datasets
data = []
for dataset_idx, dataset in enumerate(datasets):

    # Skip excluded (10 and 28 reported not having payd attention...)
    if not (int(dataset[-16:-14]) in []):
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
binarized_seqpos[data[:, 21] <= 4, 0] = 0
binarized_seqpos[data[:, 21] >= 5, 0] = 1
data = np.hstack((data, binarized_seqpos))

# Add binarized block positions
binarized_blockpos = np.zeros((data.shape[0], 1)) - 1
binarized_blockpos[data[:, 1] <= 7, 0] = 0
binarized_blockpos[data[:, 1] >= 8, 0] = 1
data = np.hstack((data, binarized_blockpos))


# 23 seqpos_binarized

# Remove non defined
data = data[data[:, 16] != 2, :]  # Remove missing resposes
data = data[data[:, 1] > 4, :]  # Remove practice blocks
data = data[data[:, 9] != -1, :]  # Remove non-defined sequences

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
    "blockpos_binarized",
]

# ==============================================================================================================

# Create df
df_root = pd.DataFrame(data=data, columns=columns)

# Select variables to consider
factors = ["bonustrial", "task_switch"]

# Calculate rates of correct responses, incorrect responses and omissions
df_rates = df_root.groupby(["id"] + factors, as_index=False).log_accuracy.value_counts(
    normalize=True
)

# Create a dara frame for response times using only correct responses
df_behavior = pd.DataFrame(data=data[data[:, 16] == 1, :], columns=columns)

# Calculate rates of correct responses, incorrect responses and omissions
df_behavior_grouped = df_behavior.groupby(["id"] + factors, as_index=False)[
    "log_rt"
].mean()

# Iterate rows
incorrect = []
correct = []
omission = []
ies = []
swibon = []
for index, row in df_behavior_grouped.iterrows():

    # Find matching rows
    df = df_rates[
        (df_rates["id"] == row["id"])
        & (df_rates[factors[0]] == row[factors[0]])
        & (df_rates[factors[1]] == row[factors[1]])
    ]

    # Get rates
    if df[df["log_accuracy"] == 0]["proportion"].to_numpy().size:
        incorrect.append(df[df["log_accuracy"] == 0]["proportion"].to_numpy()[0])
    else:
        incorrect.append(0)
    if df[df["log_accuracy"] == 1]["proportion"].to_numpy().size:
        correct.append(df[df["log_accuracy"] == 1]["proportion"].to_numpy()[0])
    else:
        correct.append(0)
    if df[df["log_accuracy"] == 2]["proportion"].to_numpy().size:
        omission.append(df[df["log_accuracy"] == 2]["proportion"].to_numpy()[0])
    else:
        omission.append(0)

# Add columns
df_behavior_grouped["incorrect"] = incorrect
df_behavior_grouped["correct"] = correct
df_behavior_grouped["omission"] = omission

# Group for anova
df_anova = df_behavior_grouped.groupby(["id"] + factors, as_index=False)[
    ["log_rt", "correct"]
].mean()

# RT analysis ===================================================================================

# Plot RTs (still including incorrect?)
g = sns.catplot(
    x=factors[0],
    y="log_rt",
    hue=factors[1],
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_behavior_grouped,
)
g.despine(left=True)

anova_out_rt = statsmodels.stats.anova.AnovaRM(
    data=df_anova, depvar="log_rt", subject="id", within=factors,
).fit()
print(anova_out_rt)


# Get values for veusz
out_data = np.zeros((2, 4))
col_counter = 0
for bon in [1, 2]:
    col_counter += 1
    for swi in [1, 2]:

        idx = (df_anova["bonustrial"] == bon - 1) & (df_anova["task_switch"] == swi - 1)
        rt_values = df_anova["log_rt"][idx].to_numpy()
        rt_m = rt_values.mean()
        rt_std = rt_values.std()

        col_offset = (col_counter - 1) * 2

        out_data[swi - 1, col_offset] = rt_m
        out_data[swi - 1, col_offset + 1] = rt_std

np.savetxt(f"{path_veusz}veusz_rt.csv", out_data, delimiter="\t")

# Accuracy analysis ===================================================================================

# Plot accuracy
g = sns.catplot(
    x=factors[0],
    y="correct",
    hue=factors[1],
    #col=factors[0],
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_behavior_grouped,
)
g.despine(left=True)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_anova, depvar="correct", subject="id", within=factors,
).fit()
print(anova_out)

# Get values for veusz
out_data = np.zeros((2, 4))
col_counter = 0
for bon in [1, 2]:
    col_counter += 1
    for swi in [1, 2]:

        idx = (df_anova["bonustrial"] == bon - 1) & (df_anova["task_switch"] == swi - 1)
        acc_values = df_anova["correct"][idx].to_numpy()
        acc_m = acc_values.mean()
        acc_std = acc_values.std()

        col_offset = (col_counter - 1) * 2

        out_data[swi - 1, col_offset] = acc_m
        out_data[swi - 1, col_offset + 1] = acc_std

np.savetxt(f"{path_veusz}veusz_accuracy.csv", out_data, delimiter="\t")

