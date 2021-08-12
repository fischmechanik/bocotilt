# Imports
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/mnt/data_dump/bocotilt/bocotilt_logs/"

# Read degree log
def read_degree_log():
    pass


# Read trial log
def read_trial_log(subject):

    # Get filename
    fn = f"{path_in}VP{subject}_trials.txt"

    # Read trials
    # Columns: 1: Trial, 2: Block, 3: Cond, 4: Resp, 5: Corr, 6: RT, 7: Bonus, 8:Task
    with open(fn) as f:
        trials = np.array([re.findall("\d+", line.rstrip()) for line in f][3:]).astype(
            np.float32
        )

    # Add bonus 1=yes 0=no
    trials = np.hstack((trials, (trials[:, 2] <= 16).astype(np.uint8).reshape(-1, 1)))

    # Add task 1=ori 0=color
    trials = np.hstack(
        (
            trials,
            (
                (np.mod(trials[:, 2], 16) > 0)
                & (np.mod(trials[:, 2], 16) < 9).astype(np.uint8)
            ).reshape(-1, 1),
        )
    )

    return trials


# Get subject list
subject_list = [file[-13:-11] for file in glob.glob(f"{path_in}/*trials.txt")]

# Read data
for subject_idx, subject in enumerate(subject_list):

    # Read trial log and add subject number
    tmp = read_trial_log(subject)
    tmp = np.hstack(
        ((np.zeros((tmp.shape[0], 1)) + int(subject)).astype(np.uint8), tmp)
    )
    if subject_idx == 0:
        trial_log = tmp
    else:
        trial_log = np.vstack((trial_log, tmp))


# Get list of blocks excluding training blocks
blocks = list(np.unique(trial_log[:, 2])[4:])

# Block averages
rt_bonus = []
rt_standard = []
for b in blocks:
    rt_bonus.append(
        trial_log[
            (trial_log[:, 5] == 1) & (trial_log[:, 2] == b) & (trial_log[:, 7] == 1), 6,
        ].mean()
    )
    rt_standard.append(
        trial_log[
            (trial_log[:, 5] == 1) & (trial_log[:, 2] == b) & (trial_log[:, 7] == 0), 6,
        ].mean()
    )

# Single trial rts
std_rts = trial_log[(trial_log[:, 5] == 1) & (trial_log[:, 7] == 0) & (trial_log[:, 2] > 4), 2]
bonus_rts = trial_log[(trial_log[:, 5] == 1) & (trial_log[:, 7] == 1) & (trial_log[:, 2] > 4), 2]
std_blocks = trial_log[(trial_log[:, 5] == 1) & (trial_log[:, 7] == 0) & (trial_log[:, 2] > 4), 6]
bonus_blocks = trial_log[(trial_log[:, 5] == 1) & (trial_log[:, 7] == 1) & (trial_log[:, 2] > 4), 6] + 0.5

# Plot
fig, ax = plt.subplots()
ax.scatter(std_rts, std_blocks, s=5, c="c", alpha=0.8)
ax.scatter(bonus_rts, bonus_blocks, s=5, c="r", alpha=0.5)
ax.plot(blocks, rt_standard, label="standard", c="c")
ax.plot(blocks, rt_bonus, label="bonus", c="r")
ax.set_xlabel("block", fontsize=10)
ax.set_ylabel("ms", fontsize=10)
ax.set_title('RTs')
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()




