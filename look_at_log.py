# Imports
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Paths
path_in = "/home/plkn/repos/bocotilt/test_logs/"

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
blocks = list(np.unique(trial_log[:, 2])[4:-1])

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
    
    
plt.plot(blocks, rt_standard)
plt.plot(blocks, rt_bonus)

