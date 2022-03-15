#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import glob
import mne
import numpy as np

# Define paths
path_in = "/mnt/data_dump/bocotilt/2_autocleaned/"


# Iterate preprocessed datasets
datasets = glob.glob(f"{path_in}/*cleaned.set")

# Still testing
datasets = [datasets[0]]

# Loop datasets
for dataset_idx, dataset in enumerate(datasets):
    
    # Get subject id as string
    id_string = dataset.split("VP")[1][0:2]

    # Load eeg data
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))
    
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
    
    # Create event codes
    combined_codes = np.array(
        [
            int(
                str(int(trialinfo[x, 3])) # Bonustrial
                + str(int(trialinfo[x, 21])) # sequence position
            )
            for x in range(trialinfo.shape[0])
        ]
    )