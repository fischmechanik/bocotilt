# Imports
import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt

# Path
path_in = "/mnt/data_dump/bocotilt/5_regression/"

# Load tf parameters
tf_times = joblib.load(f"{path_in}tf_times.joblib")
tf_freqs = joblib.load(f"{path_in}tf_freqs.joblib")


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
    coefs_standard_trialnum.append(data["coef_standard_trialnum"])
    coefs_standard_seqpos.append(data["coef_standard_seqpos"])
    coefs_bonus_trialnum.append(data["coef_bonus_trialnum"])
    coefs_bonus_seqpos.append(data["coef_bonus_seqpos"])

# Stack
coefs_standard_trialnum = np.stack(coefs_standard_trialnum)
coefs_standard_seqpos = np.stack(coefs_standard_seqpos)
coefs_bonus_trialnum = np.stack(coefs_bonus_trialnum)
coefs_bonus_seqpos = np.stack(coefs_bonus_seqpos)

maxval = 0.05
plotdata = coefs_standard_seqpos.mean(axis=0)[126]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

plotdata = coefs_standard_trialnum.mean(axis=0)[126]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

maxval = 0.05
plotdata = coefs_bonus_seqpos.mean(axis=0)[126]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")

plotdata = coefs_bonus_trialnum.mean(axis=0)[126]
plt.contourf(tf_times, tf_freqs, plotdata, vmin=-maxval, vmax=maxval, cmap="jet")