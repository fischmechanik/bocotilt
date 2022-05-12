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
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_theta = "/mnt/data_dump/bocotilt/7_ged/"

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
data = data[data[:, 16] != 2, :] # Remove missing resposes
data = data[data[:, 1] > 4, :]  # Remove practice blocks


data = data[data[:, 9] != -1, :] # Remove non-defined sequences

#data = data[data[:, 23] >= 0, :] # Remove trials not belonging to bnarized sequence ranges
data = data[data[:, 24] >= 0, :] # Remove trials not belonging to bnarized block ranges

# Seperate datasets for rt and accuracy analyses
data_rt = data[data[:, 16] == 1, :]  # Remove incorrect trials


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
factors = ["block_nr", "bonustrial", "task_switch"]

# Calculate rates of correct responses, incorrect responses and omissions
df_rates = df_root.groupby(
    ["id"] + factors, as_index=False
).log_accuracy.value_counts(normalize=True)

# Create a dara frame for response times using only correct responses
df_behavior = pd.DataFrame(data=data[data[:, 16] == 1, :], columns=columns)

# Calculate rates of correct responses, incorrect responses and omissions
df_behavior_grouped = df_behavior.groupby(
   ["id"] + factors, as_index=False
)["log_rt"].mean()

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

# Add inverse efficiency score
df_behavior_grouped["ies"] = df_behavior_grouped.apply(lambda row: row["log_rt"] / row["correct"], axis=1)

# Add combined bonus-switch factor
df_behavior_grouped["ies"] = df_behavior_grouped.apply(lambda row: row["task_switch"] + row["correct"], axis=1)

g = sns.catplot(
    x=factors[0],
    y="log_rt",
    hue=factors[1],
    col=factors[2],
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_behavior_grouped,
)
g.despine(left=True)

g = sns.catplot(
    x=factors[0],
    y="correct",
    hue=factors[2],
    col=factors[1],
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_behavior_grouped,
)
g.despine(left=True)

# Sequential_position x bonus_trial for accuracies
df_anova = df_behavior_grouped.groupby(
    ["id"] + factors, as_index=False
)[["log_rt", "correct", "ies"]].mean()

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_anova,
    depvar="log_rt",
    subject="id",
    within=factors,
).fit()
print(anova_out)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_anova,
    depvar="correct",
    subject="id",
    within=factors,
).fit()
print(anova_out)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_anova,
    depvar="ies",
    subject="id",
    within=factors,
).fit()
print(anova_out)


# Run LMER
md = smf.mixedlm("log_rt ~ C(bonustrial) * C(task_switch) * block_nr", df_behavior_grouped, groups=df_behavior_grouped["id"], re_formula="~block_nr")
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())

md = smf.mixedlm("correct ~ C(bonustrial) * C(task_switch) * block_nr", df_behavior_grouped, groups=df_behavior_grouped["id"], re_formula="~block_nr")
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())

md = smf.mixedlm("ies ~ C(bonustrial) * C(task_switch) * block_nr", df_behavior_grouped, groups=df_behavior_grouped["id"], re_formula="~block_nr")
mdf = md.fit(method=["lbfgs"])
print(mdf.summary())


# Load theta table
data_theta = scipy.io.loadmat(os.path.join(path_theta, "theta_table.mat"))["theta_table"]


# Create df
columns_theta = ["id", "tot", "bonus", "switch", "theta1", "theta2"]
df_theta = pd.DataFrame(data=data_theta, columns=columns_theta)

anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta,
    depvar="theta1",
    subject="id",
    within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

g = sns.catplot(
    x="tot",
    y="theta1",
    hue="bonus",
    col="switch",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)


anova_out = statsmodels.stats.anova.AnovaRM(
    data=df_theta,
    depvar="theta2",
    subject="id",
    within=["tot", "bonus", "switch"],
).fit()
print(anova_out)

g = sns.catplot(
    x="tot",
    y="theta2",
    hue="bonus",
    col="switch",
    capsize=0.05,
    palette="tab20",
    height=6,
    aspect=0.75,
    kind="point",
    data=df_theta,
)
g.despine(left=True)






























