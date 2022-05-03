# Imports
import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt
import scipy.io
import os
import mne

# Path
path_clean_data = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_in = "/mnt/data_dump/bocotilt/5_regression/"

# Load tf parameters
tf_times = joblib.load(f"{path_in}tf_times.joblib")
tf_freqs = joblib.load(f"{path_in}tf_freqs.joblib")

# Load channel labels
channel_labels = str(
    scipy.io.loadmat(os.path.join(path_clean_data, "channel_labels.mat"))["channel_labels"]
)[3:-2].split(" ")

# Replace O9 and O10 with I1 and I2
channel_labels = ['I1' if ch == 'O9' else ch for ch in channel_labels]
channel_labels = ['I2' if ch == 'O10' else ch for ch in channel_labels]

# Create a basic mne info structure
sfreq = 100
info = mne.create_info(channel_labels, sfreq, ch_types="eeg", verbose=None)

# Create a montage
standard_montage = mne.channels.make_standard_montage("standard_1005")

# Create mne epochs objects
baseline = (-0.2, 0)

dummy_epochs = mne.EpochsArray(
    np.zeros((10, len(channel_labels), len(tf_times))), info
)
dummy_epochs.set_montage(standard_montage)
    
# Define channel adjacency matrix
adjacency, channel_labels = mne.channels.find_ch_adjacency(dummy_epochs.info, ch_type=None)

# Define adjacency in time and frq domain as well...
tf_adjacency = mne.stats.combine_adjacency(len(tf_freqs), len(tf_times), adjacency)

# Get datasets
datasets = glob.glob(f"{path_in}/*_regression_data.joblib")

# Coef lists
coefs_standard_trialnum = []
coefs_standard_seqpos = []
coefs_bonus_trialnum = []
coefs_bonus_seqpos = []

# Load coefs
for dataset_idx, dataset in enumerate(datasets):
    data = joblib.load(dataset)
    coefs_standard_trialnum.append(data["coef_standard_trialnum"].transpose((1,2,0)))
    coefs_standard_seqpos.append(data["coef_standard_seqpos"].transpose((1,2,0)))
    coefs_bonus_trialnum.append(data["coef_bonus_trialnum"].transpose((1,2,0)))
    coefs_bonus_seqpos.append(data["coef_bonus_seqpos"].transpose((1,2,0)))

# Stack
coefs_standard_trialnum = np.stack(coefs_standard_trialnum)
coefs_standard_seqpos = np.stack(coefs_standard_seqpos)
coefs_bonus_trialnum = np.stack(coefs_bonus_trialnum)
coefs_bonus_seqpos = np.stack(coefs_bonus_seqpos)

# Cluster test
F_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_test(
    [coefs_standard_seqpos, coefs_bonus_seqpos],
    threshold=None,
    n_permutations=1024,
    tail=0,
    stat_fun=None,
    adjacency=tf_adjacency,
    n_jobs=-2,
    seed=None,
    max_step=1,
    exclude=None,
    step_down_p=0,
    t_power=1,
    out_type="indices",
    check_disjoint=False,
    buffer_size=1000,
    verbose=None,
)

aa=bb
maxval = 0.2

plotdata = coefs_standard_seqpos.mean(axis=0)[4]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

plotdata = coefs_standard_trialnum.mean(axis=0)[4]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

plotdata = coefs_bonus_seqpos.mean(axis=0)[4]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

plotdata = coefs_bonus_trialnum.mean(axis=0)[4]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")