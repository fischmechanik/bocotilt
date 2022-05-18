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
path_in = "/home/plkn/bocotilt_ged/2_autocleaned/"
path_theta = "/home/plkn/bocotilt_ged/7_ged/"
path_veusz = "/home/plkn/bocotilt_ged/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Read datasets
data = []
for dataset_idx, dataset in enumerate(datasets):

    # Skip excluded
    if not (int(dataset[-16:-14]) in [10, 28]):
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

# data = data[data[:, 23] >= 0, :] # Remove trials not belonging to bnarized sequence ranges
data = data[data[:, 24] >= 0, :]  # Remove trials not belonging to bnarized block ranges

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
factors = ["blockpos_binarized", "bonustrial", "task_switch"]

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
        & (df_rates[factors[2]] == row[factors[2]])
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

    if (row["bonustrial"] == 0) & (row["task_switch"] == 0):
        swibon.append(0)
    if (row["bonustrial"] == 0) & (row["task_switch"] == 1):
        swibon.append(1)
    if (row["bonustrial"] == 1) & (row["task_switch"] == 0):
        swibon.append(2)
    if (row["bonustrial"] == 1) & (row["task_switch"] == 1):
        swibon.append(3)

# Add columns
df_behavior_grouped["incorrect"] = incorrect
df_behavior_grouped["correct"] = correct
df_behavior_grouped["omission"] = omission
df_behavior_grouped["swibon"] = swibon

# Group for anova
df_anova = df_behavior_grouped.groupby(["id"] + factors, as_index=False)[
    ["log_rt", "correct"]
].mean()

# RT analysis ===================================================================================

# Plot RTs (still including incorrect?)
g = sns.catplot(
    x=factors[2],
    y="log_rt",
    hue=factors[1],
    col=factors[0],
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_behavior_grouped,
)
g.despine(left=True)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_anova, depvar="log_rt", subject="id", within=factors,
).fit()
print(anova_out)

df_start_only = (
    df_anova[df_anova["blockpos_binarized"] == 0]
    .groupby(["id", "bonustrial", "task_switch"], as_index=False)[["log_rt", "correct"]]
    .mean()
)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_start_only,
    depvar="log_rt",
    subject="id",
    within=["bonustrial", "task_switch"],
).fit()
print(anova_out)

df_end_only = (
    df_anova[df_anova["blockpos_binarized"] == 1]
    .groupby(["id", "bonustrial", "task_switch"], as_index=False)[["log_rt", "correct"]]
    .mean()
)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_end_only,
    depvar="log_rt",
    subject="id",
    within=["bonustrial", "task_switch"],
).fit()
print(anova_out)

# Get values for veusz
veusz_rt = np.zeros((4, 4))
for tot in [0, 1]:
    row_counter = -1
    for bon in [0, 1]:
        for swi in [0, 1]:

            row_counter += 1

            idx = (
                (df_anova["blockpos_binarized"] == tot)
                & (df_anova["bonustrial"] == bon)
                & (df_anova["task_switch"] == swi)
            )
            rts = df_anova["log_rt"][idx].to_numpy()
            rt_m = rts.mean()
            rt_std = rts.std()

            col_offset = tot * 2

            veusz_rt[row_counter, col_offset] = rt_m
            veusz_rt[row_counter, col_offset + 1] = rt_std

np.savetxt(f"{path_veusz}veusz_rt.csv", veusz_rt, delimiter="\t")

np.savetxt(f"{path_veusz}xax.csv", [1, 2, 3, 4], delimiter="\t")

# Accuracy analysis ===================================================================================

# Plot accuracy
g = sns.catplot(
    x=factors[2],
    y="correct",
    hue=factors[1],
    col=factors[0],
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
veusz_accuracy = np.zeros((4, 4))
for tot in [0, 1]:
    row_counter = -1
    for bon in [0, 1]:
        for swi in [0, 1]:

            row_counter += 1

            idx = (
                (df_anova["blockpos_binarized"] == tot)
                & (df_anova["bonustrial"] == bon)
                & (df_anova["task_switch"] == swi)
            )
            acc_values = df_anova["correct"][idx].to_numpy()
            acc_m = acc_values.mean()
            acc_std = acc_values.std()

            col_offset = tot * 2

            veusz_accuracy[row_counter, col_offset] = acc_m
            veusz_accuracy[row_counter, col_offset + 1] = acc_std

np.savetxt(f"{path_veusz}veusz_accuracy.csv", veusz_accuracy, delimiter="\t")


# LME analysis ===================================================================================


# # Calculate rates of correct responses, incorrect responses and omissions
# df_behavior_grouped_block = df_behavior.groupby(
#    ["id", "bonustrial", "task_switch", "block_nr"], as_index=False
# )["log_rt", "log_accuracy"].mean()

# # Plot accuracy
# g = sns.catplot(
#     x="block_nr",
#     y="log_rt",
#     hue="bonustrial",
#     col="task_switch",
#     capsize=0.05,
#     palette="tab20",
#     height=6,
#     aspect=0.75,
#     kind="point",
#     data=df_behavior_grouped_block,
# )
# g.despine(left=True)

# # Run LMER
# md = smf.mixedlm("log_rt ~ C(bonustrial) * C(task_switch) * block_nr", df_behavior_grouped_block, groups=df_behavior_grouped_block["id"], re_formula="~block_nr")
# mdf = md.fit(method=["lbfgs"])
# print(mdf.summary())


# Load theta table
data_theta = scipy.io.loadmat(os.path.join(path_theta, "theta_table.mat"))[
    "theta_table"
]

# Create df
columns_theta = [
    "id",
    "tot",
    "bonus",
    "switch",
    "ersp_win_1",
    "ersp_win_2",
    "itpc_win_1",
    "itpc_win_2",
    "rt",
    "acc",
]
df_theta = pd.DataFrame(data=data_theta, columns=columns_theta)


# ERSP analysis =================================================================================

# Plot theta
g = sns.catplot(
    x="switch",
    y="ersp_win_1",
    hue="bonus",
    col="tot",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)

# First timewin
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta, depvar="ersp_win_1", subject="id", within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

# Get values for veusz
veusz_theta1 = np.zeros((4, 4))
for tot in [1, 2]:
    row_counter = -1
    for bon in [1, 2]:
        for swi in [1, 2]:

            row_counter += 1

            idx = (
                (df_theta["tot"] == tot)
                & (df_theta["bonus"] == bon)
                & (df_theta["switch"] == swi)
            )
            theta1_values = df_theta["ersp_win_1"][idx].to_numpy()
            theta1_m = theta1_values.mean()
            theta1_std = theta1_values.std()

            col_offset = (tot - 1) * 2

            veusz_theta1[row_counter, col_offset] = theta1_m
            veusz_theta1[row_counter, col_offset + 1] = theta1_std

np.savetxt(f"{path_veusz}veusz_theta_1.csv", veusz_theta1, delimiter=",")

# Plot theta
g = sns.catplot(
    x="switch",
    y="ersp_win_2",
    hue="bonus",
    col="tot",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)

# First timewin
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta, depvar="ersp_win_2", subject="id", within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

# Get values for veusz
veusz_theta2 = np.zeros((4, 4))
for tot in [1, 2]:
    row_counter = -1
    for bon in [1, 2]:
        for swi in [1, 2]:

            row_counter += 1

            idx = (
                (df_theta["tot"] == tot)
                & (df_theta["bonus"] == bon)
                & (df_theta["switch"] == swi)
            )
            theta2_values = df_theta["ersp_win_2"][idx].to_numpy()
            theta2_m = theta2_values.mean()
            theta2_std = theta2_values.std()

            col_offset = (tot - 1) * 2

            veusz_theta2[row_counter, col_offset] = theta2_m
            veusz_theta2[row_counter, col_offset + 1] = theta2_std

np.savetxt(f"{path_veusz}veusz_theta_2.csv", veusz_theta2, delimiter=",")

# ITPC analysis =================================================================================

# Plot 
g = sns.catplot(
    x="switch",
    y="itpc_win_1",
    hue="bonus",
    col="tot",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)

# First timewin
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta, depvar="itpc_win_1", subject="id", within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

# Plot
g = sns.catplot(
    x="switch",
    y="itpc_win_2",
    hue="bonus",
    col="tot",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)

# First timewin
anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta, depvar="itpc_win_2", subject="id", within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

