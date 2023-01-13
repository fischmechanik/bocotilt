#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import sys
import statsmodels.stats.anova

# Define paths
path_in = "/mnt/data_dump/bocotilt/8_fooof/fooof_models/"
path_fooof = "/home/plkn/Downloads/fooof/"

# Append fooof to sys path
sys.path.append(path_fooof)

# Import fooof
import fooof

# List of datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Get freqs
freqs = joblib.load(datasets[0])["baseline"][0].freqs

# Init pandas stuff
cols = [
    "id",
    "condition",
    "reward",
    "switch",
    "timewin",
    "freq",
    "pow",
]
df = pd.DataFrame(columns=cols, index=range(len(datasets) * 12 * len(freqs)))
df_idx_counter = -1

# Loop datasets
for counter_subject, dataset in enumerate(datasets):

    # Load dataset
    fooof_data = joblib.load(dataset)

    # Get condition idx
    tinf = fooof_data["trialinfo"]

    condition_idx = [
        tinf.index[(tinf["bonus"] == 0) & (tinf["task_switch"] == 0)].tolist(),
        tinf.index[(tinf["bonus"] == 0) & (tinf["task_switch"] == 1)].tolist(),
        tinf.index[(tinf["bonus"] == 1) & (tinf["task_switch"] == 0)].tolist(),
        tinf.index[(tinf["bonus"] == 1) & (tinf["task_switch"] == 1)].tolist(),
    ]

    # Condition labels
    condition_labels = ["std_rep", "std_swi", "bon_rep", "bon_swi"]

    # Loop conditions
    for condition_nr, cidx in enumerate(condition_idx):

        # Iterate time windows
        for tw_idx, tw in enumerate(["baseline", "ct_interval", "post_target"]):

            # Get a more compact time window identifier
            tw_compact = ["bl", "ct", "pt"][tw_idx]

            # Parameter matrix
            spectrum = np.zeros(fooof_data["baseline"][0].power_spectrum.shape[0])

            # Loop trials of condition
            for idx in cidx:

                # Accumulate 
                spectrum += fooof_data[tw][idx].fooofed_spectrum_
                
            # Adjust
            spectrum = spectrum / len(cidx)
            
            # Populate
            for freq_idx, freq in enumerate(freqs):
                df_idx_counter += 1
                df.loc[df_idx_counter]["id"] = int(dataset.split("/")[-1][:2])
                df.loc[df_idx_counter]["condition"] = condition_labels[condition_nr]
                df.loc[df_idx_counter]["reward"] = condition_labels[condition_nr][0:3]
                df.loc[df_idx_counter]["switch"] = condition_labels[condition_nr][4:7]
                df.loc[df_idx_counter]["timewin"] = tw
                df.loc[df_idx_counter]["freq"] = freq
                df.loc[df_idx_counter]["pow"] = spectrum[freq_idx]
                
# Plot              
sns.lineplot(data=df, x="freq", y="pow", hue="condition", style="timewin")
                    
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
