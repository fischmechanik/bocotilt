#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np
import itertools
from cool_colormaps import cga_p1_dark as ccm

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"
path_out = "/mnt/data_dump/bocotilt/5_regression/"

# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")

# Create a montage
standard_montage = mne.channels.make_standard_montage("standard_1005")

# Loop datasets
power_list = []
itc_list = []
condition_labels = []
for dataset_idx, dataset in enumerate(datasets):

    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Rename channels and seet montage
    eeg_epochs.rename_channels({"O9": "I1", "O10": "I2"})
    eeg_epochs.set_montage(standard_montage)

    # Load trialinfo
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
    trialinfo = np.genfromtxt(
        dataset.split("VP")[0] + "VP" + id_string + "_trialinfo.csv", delimiter=","
    )

    # Exclude practice blocks
    eeg_epochs.drop(trialinfo[:, 1] <= 4)
    trialinfo = trialinfo[trialinfo[:, 1] > 4, :]

    # Define experimental factors and corresponding trialinfo column
    factors = {"block": 1, "bonus": 3}

    # Get all factor level combinations
    factor_levels = [np.unique(trialinfo[:, factors[f]]) for f in factors]
    factor_level_combinations = list(itertools.product(*factor_levels))

    # Loop factor level combinations
    for comb in factor_level_combinations:

        # A condition label
        condition_label = ""
        
        # Loop factors and get boolean mask for epoch selection
        bool_mask = np.ones((trialinfo.shape[0],), dtype=bool)
        for factor_nr, level in enumerate(comb):

            # Get key of factor
            key = list(factors.keys())[factor_nr]
            
            # Update condition label
            condition_label = condition_label + key + str(int(level)) + "_"

            # Update bool mask
            bool_mask = bool_mask & (trialinfo[:, factors[key]] == level)
            
        

        # Select epochs
        eeg_epochs_comb = eeg_epochs[bool_mask]

        # Perform time-frequency analysis and apply baseline
        tf_freqs = np.linspace(2, 16, 20)
        tf_cycles = np.linspace(3, 12, 20)
        power, itc = mne.time_frequency.tfr_morlet(
            eeg_epochs, tf_freqs, n_cycles=tf_cycles, n_jobs=-2, decim=2,
        )
        power.apply_baseline(mode="logratio", baseline=(-0.5, -0.2))

    # Collect
    power_list.append(power)
    itc_list.append(itc)


# Grand averages
ga = mne.grand_average(power_list)


# Plotty
ga.plot_joint(
    tmin=-0.5, tmax=2, timefreqs=[(0.5, 10), (1.1, 4)], cmap="PuOr", vmin=-0.3, vmax=0.3
)

aa = bb

