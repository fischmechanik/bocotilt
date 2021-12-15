% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';

% Subject list
subject_list = {'VP08'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Loop subjects
for s = 1 : length(subject_list)

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Trial data
    % Columns:
    %  1: id
    %  2: block_nr
    %  3: trial_nr
    %  4: bonustrial
    %  5: tilt_task
    %  6: cue_ax
    %  7: target_red_left
    %  8: distractor_red_left
    %  9: response_interference
    % 10: task_switch
    % 11: correct_response
    % 12: response_side
    % 13: rt
    % 14: accuracy
    % 15: log_response_side
    % 16: log_rt
    % 17: log_accuracy
    % 18: position_color
    % 19: position_tilt
    % 20: position_target
    % 21: position_distractor
    % 22: sequence_position
    % 23: sequence_length
    trialinfo = EEG.trialinfo;

    % Time vector
    erp_times = EEG.times;
    
    % The data matrix: channels x times x trials
    eeg_data = EEG.data;

    % Get index of channel
    channel = 'Fz';
    channel_idx = find(strcmp({EEG.chanlocs.labels}, channel));

    % Get indices of correct standard and bonus trials
    idx_std_rep = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 10) == 0;
    idx_std_swi = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 10) == 1;
    idx_bon_rep = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 10) == 0;
    idx_bon_swi = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 10) == 1;

    % Get ERPs
    erp_std_rep = mean(squeeze(eeg_data(channel_idx, :, idx_std_rep)), 2);
    erp_std_swi = mean(squeeze(eeg_data(channel_idx, :, idx_std_swi)), 2);
    erp_bon_rep = mean(squeeze(eeg_data(channel_idx, :, idx_bon_rep)), 2);
    erp_bon_swi = mean(squeeze(eeg_data(channel_idx, :, idx_bon_swi)), 2);


    % Plot
    figure;
    plot(erp_times, erp_std_rep, '-', 'LineWidth', 1.5, 'Color', 'red');
    hold on;
    plot(erp_times, erp_std_swi, '--', 'LineWidth', 1.5, 'Color', 'red');
    plot(erp_times, erp_bon_rep, '-', 'LineWidth', 1.5, 'Color', 'black');
    plot(erp_times, erp_bon_swi, '--', 'LineWidth', 1.5, 'Color', 'black');
    legend({'standard-repeat', 'standard-switch', 'bonus-repeat', 'bonus-switch'});

end