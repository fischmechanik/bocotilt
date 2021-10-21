# Imports
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/mnt/data2/bocotilt/3_autocleaned/"

# Get datasets
datasets = glob.glob(f"{path_in}/*_trialinfo.csv")

# Read datasets
for dataset_idx, dataset in enumerate(datasets):
    if dataset_idx == 0:
        dat = np.genfromtxt(dataset, delimiter=",")
    else:
        dat = np.stack((dat, np.genfromtxt(dataset, delimiter=",")))

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
# 14: sequence_position
# 15: sequence_length
# 16: ordered
# 17: position_color
# 18: position_tilt
# 19: position_target
# 20: position_distractor

rt_rnd_bon = dat[(dat[:, 3] == 1) & (dat[:, 16] == 0), 12]
rt_rnd_std = dat[(dat[:, 3] == 0) & (dat[:, 16] == 0), 12]
rt_ord_bon = dat[(dat[:, 3] == 1) & (dat[:, 16] == 1), 12]
rt_ord_std = dat[(dat[:, 3] == 0) & (dat[:, 16] == 1), 12]

mean_rt = np.array(
    [
        np.nanmean(rt_rnd_bon),
        np.nanmean(rt_rnd_std),
        np.nanmean(rt_ord_bon),
        np.nanmean(rt_ord_std),
    ]
)
x = np.arange(4)

plt.bar(x, mean_rt)
