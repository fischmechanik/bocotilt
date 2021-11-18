# Imports
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Get a single dataset for now...
datasets = glob.glob(f"{path_in}/VP08_trialinfo.csv")

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

# Get RTs
bon_pos1_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 12]
)
std_pos1_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 12]
)
bon_pos2_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 12]
)
std_pos2_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 12]
)
bon_pos3_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 12]
)
std_pos3_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 12]
)
bon_pos4_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 12]
)
std_pos4_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 12]
)
bon_pos5_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 12]
)
std_pos5_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 12]
)
bon_pos6_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 12]
)
std_pos6_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 12]
)
bon_pos7_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 12]
)
std_pos7_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 12]
)
bon_pos8_mean = np.nanmean(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 12]
)
std_pos8_mean = np.nanmean(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 12]
)
bon_pos1_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 12]
)
std_pos1_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 1) & (dat[:, 1] > 4), 12]
)
bon_pos2_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 12]
)
std_pos2_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 2) & (dat[:, 1] > 4), 12]
)
bon_pos3_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 12]
)
std_pos3_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 3) & (dat[:, 1] > 4), 12]
)
bon_pos4_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 12]
)
std_pos4_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 4) & (dat[:, 1] > 4), 12]
)
bon_pos5_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 12]
)
std_pos5_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 5) & (dat[:, 1] > 4), 12]
)
bon_pos6_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 12]
)
std_pos6_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 6) & (dat[:, 1] > 4), 12]
)
bon_pos7_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 12]
)
std_pos7_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 7) & (dat[:, 1] > 4), 12]
)
bon_pos8_error = np.nanstd(
    dat[(dat[:, 3] == 1) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 12]
)
std_pos8_error = np.nanstd(
    dat[(dat[:, 3] == 0) & (dat[:, 18] == 8) & (dat[:, 1] > 4), 12]
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
bon_error = [
    bon_pos1_error,
    bon_pos2_error,
    bon_pos3_error,
    bon_pos4_error,
    bon_pos5_error,
    bon_pos6_error,
    bon_pos7_error,
    bon_pos8_error,
]
std_error = [
    std_pos1_error,
    std_pos2_error,
    std_pos3_error,
    std_pos4_error,
    std_pos5_error,
    std_pos6_error,
    std_pos7_error,
    std_pos8_error,
]


# Build the plot
fig, ax = plt.subplots()

rects1 = ax.bar(x_pos - barwidth / 2, std_means, barwidth, label="standard")
rects2 = ax.bar(x_pos + barwidth / 2, bon_means, barwidth, label="bonus")

ax.set_ylabel("ms")
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_labels)
ax.set_title("mean RT standard vs bonus trials")
ax.yaxis.grid(True)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

