# Imports
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Read datasets
data = []
for dataset_idx, dataset in enumerate(datasets):
    data.append(np.genfromtxt(dataset, delimiter=","))

# Get subject data
dat = data[4]

# get id
subject_id = int(dat[0, 0])

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
# 14: log_response_side
# 15: log_rt
# 16: log_accuracy
# 17: position_color
# 18: position_tilt
# 19: position_target
# 20: position_distractor
# 21: sequence_position
# 22: sequence_length

# Get RT
col_rt = 15
col_accuracy = 16
col_blocknum = 1
col_bonustrial = 3
col_seqposition = 21
bon_pos1_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos1_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos2_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos2_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos3_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos3_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos4_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos4_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos5_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos5_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos6_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos6_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos7_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos7_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
bon_pos8_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
)
std_pos8_mean = np.nanmean(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4)
        & (dat[:, col_accuracy] == 1),
        col_rt,
    ]
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
ax[0].set_ylim((0, 900))
ax[0].set_xticks(x_pos)
ax[0].set_xticklabels(condition_labels)
ax[0].set_title(f"subject {subject_id} - RT")
ax[0].yaxis.grid(True)
ax[0].legend()

# Get accuracy
bon_pos1_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos1_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 1)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos2_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos2_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 2)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos3_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos3_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 3)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos4_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos4_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 4)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos5_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos5_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 5)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos6_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos6_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 6)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos7_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos7_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 7)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
bon_pos8_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 1)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)
std_pos8_acc = sum(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
    == 1
) / len(
    dat[
        (dat[:, col_bonustrial] == 0)
        & (dat[:, col_seqposition] == 8)
        & (dat[:, col_blocknum] > 4),
        col_accuracy,
    ]
)

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
ax[1].set_ylim((0.3, 1))
ax[1].set_xticks(x_pos)
ax[1].set_xticklabels(condition_labels)
ax[1].set_title("accuracy")
ax[1].yaxis.grid(True)


fig.tight_layout()
plt.show()





















