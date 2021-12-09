# Imports
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Get a single dataset for now...
datasets = glob.glob(f"{path_in}/VP09_trialinfo.csv")

# Read datasets
data = []
for dataset_idx, dataset in enumerate(datasets):
    data.append(np.genfromtxt(dataset, delimiter=","))

dat = data[0]

# Columns of dat
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
# 14: position_color
# 15: position_tilt
# 16: position_target
# 17: position_distractor
# 18: sequence_position
# 19: sequence_length

# Get RT
bon_pos1_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 1) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos1_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 1) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos2_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 2) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos2_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 2) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos3_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 3) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos3_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 3) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos4_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 4) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos4_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 4) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos5_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 5) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos5_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 5) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos6_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 6) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos6_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 6) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos7_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 7) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos7_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 7) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
bon_pos8_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 8) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)
std_pos8_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 8) & (dat[:, 1] > 4) & (dat[:, 13] == 1), 12]
)

# Arange plotting data
condition_labels = [
    "pos1",
    "pos2",
    "pos3",
    "pos4",
    "pos5",
    "pos6",
    "pos7",
    "pos8",
]
x_pos = np.arange(len(condition_labels))
barwidth = 0.35
bon_means = [
    bon_pos1_mean,
    bon_pos2_mean,
    bon_pos3_mean,
    bon_pos4_mean,
    bon_pos5_mean,
    bon_pos6_mean,
    bon_pos7_mean,
    bon_pos8_mean,
]
std_means = [
    std_pos1_mean,
    std_pos2_mean,
    std_pos3_mean,
    std_pos4_mean,
    std_pos5_mean,
    std_pos6_mean,
    std_pos7_mean,
    std_pos8_mean,
]


# Build the plot
fig, ax = plt.subplots(2, 1)
rects1 = ax[0].bar(
    x_pos - barwidth / 2, std_means, barwidth, label="standard", color="olive"
)
rects2 = ax[0].bar(
    x_pos + barwidth / 2, bon_means, barwidth, label="bonus", color="salmon"
)
ax[0].set_ylabel("ms")
ax[0].set_ylim((500, 900))
ax[0].set_xticks(x_pos)
ax[0].set_xticklabels(condition_labels)
ax[0].set_title("RT")
ax[0].yaxis.grid(True)
ax[0].legend()

# Get accuracy
bon_pos1_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 13])
std_pos1_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 13])
bon_pos2_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 13])
std_pos2_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 13])
bon_pos3_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 13])
std_pos3_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 13])
bon_pos4_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 13])
std_pos4_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 13])
bon_pos5_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 13])
std_pos5_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 13])
bon_pos6_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 13])
std_pos6_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 13])
bon_pos7_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 13])
std_pos7_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 13])
bon_pos8_acc = sum(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 1) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 13])
std_pos8_acc = sum(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 13] == 1
) / len(dat[(dat[:, 3] == 0) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 13])

# Arange plotting data
barwidth = 0.35
bon_acc = [
    bon_pos1_acc,
    bon_pos2_acc,
    bon_pos3_acc,
    bon_pos4_acc,
    bon_pos5_acc,
    bon_pos6_acc,
    bon_pos7_acc,
    bon_pos8_acc,
]
std_acc = [
    std_pos1_acc,
    std_pos2_acc,
    std_pos3_acc,
    std_pos4_acc,
    std_pos5_acc,
    std_pos6_acc,
    std_pos7_acc,
    std_pos8_acc,
]

# Build the plot
rects1 = ax[1].bar(
    x_pos - barwidth / 2, std_acc, barwidth, label="standard", color="olive"
)
rects2 = ax[1].bar(
    x_pos + barwidth / 2, bon_acc, barwidth, label="bonus", color="salmon"
)
ax[1].set_ylabel("% correct")
ax[1].set_ylim((0.5, 0.9))
ax[1].set_xticks(x_pos)
ax[1].set_xticklabels(condition_labels)
ax[1].set_title("accuracy")
ax[1].yaxis.grid(True)


fig.tight_layout()
plt.show()

