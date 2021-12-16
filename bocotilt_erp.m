% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';

% Subject list
subject_list = {'VP08', 'VP09', 'VP17'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% This is where we collect the ERPs as a subject x condition x times matrix
erp_matrix = [];

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

    % Workaround...
    trialinfo(trialinfo(:, 22) == 1, 10) = -1;

    % Time vector
    erp_times = EEG.times;
    
    % The data matrix: channels x times x trials
    eeg_data = EEG.data;

    % Get index of channel
    channel_idx = [];
    %channels = {'Fz', 'F1', 'F2', 'FC1', 'FC2', 'FFC1h', 'FFC2h'};
    %channels = {'Pz', 'POz', 'PPO1h', 'PPO2h'};
    channels = {'POz'};
    for ch = 1 : length(channels)
        channel_idx(end + 1) = find(strcmp({EEG.chanlocs.labels}, channels{ch}));
    end

    % Average data across selected channels (result is 2d matrix, times x trials)
    chan_data = squeeze(mean(eeg_data(channel_idx, :, :), 1));

    % Define baseline
    bl_win = [-200, 0];
    bl_idx = erp_times >= bl_win(1) & erp_times <= bl_win(2);
    chan_data = chan_data - mean(chan_data(bl_idx, :));

    % Get indices of correct standard and bonus trials for repetition and switch trials
    idx_std_rep = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 10) == 0;
    idx_std_swi = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 10) == 1;
    idx_bon_rep = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 10) == 0;
    idx_bon_swi = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 10) == 1;

    % Calculate ERPs by averaging across trials within each condition combination
    erp_std_rep = mean(chan_data(:, idx_std_rep), 2);
    erp_std_swi = mean(chan_data(:, idx_std_swi), 2);
    erp_bon_rep = mean(chan_data(:, idx_bon_rep), 2);
    erp_bon_swi = mean(chan_data(:, idx_bon_swi), 2);

    % Get indices of correct standard and bonus trials for start- versus end-of-sequence trials
    idx_std_start = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 22) <= 4;
    idx_std_end   = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 0 & trialinfo(:, 22) > 4;
    idx_bon_start = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 22) <= 4;
    idx_bon_end   = trialinfo(:, 17) == 1 & trialinfo(:, 2) > 4 & trialinfo(:, 4) == 1 & trialinfo(:, 22) > 4;

    % Calculate ERPs by averaging across trials within each condition combination
    erp_std_start = mean(chan_data(:, idx_std_start), 2);
    erp_std_end   = mean(chan_data(:, idx_std_end  ), 2);
    erp_bon_start = mean(chan_data(:, idx_bon_start), 2);
    erp_bon_end   = mean(chan_data(:, idx_bon_end  ), 2);

    % Copy ERPs to ERP-result matrix
    erp_matrix(s, 1, :) = erp_std_rep;
    erp_matrix(s, 2, :) = erp_std_swi;
    erp_matrix(s, 3, :) = erp_bon_rep;
    erp_matrix(s, 4, :) = erp_bon_swi;

    erp_matrix(s, 5, :) = erp_std_start;
    erp_matrix(s, 6, :) = erp_std_end;
    erp_matrix(s, 7, :) = erp_bon_start;
    erp_matrix(s, 8, :) = erp_bon_end;

end % End subject loop

% Average ERPs across subjects -> grand averages as condition x time matrix 
grand_averages = squeeze(mean(erp_matrix, 1));

% Create a plot of ERPs, averaged across subjects
figure;
plot(erp_times, grand_averages(1, :), '-', 'LineWidth', 2, 'Color', 'k');
hold on;
plot(erp_times, grand_averages(2, :), ':', 'LineWidth', 2, 'Color', 'k');
plot(erp_times, grand_averages(3, :), '-', 'LineWidth', 2, 'Color', 'r');
plot(erp_times, grand_averages(4, :), ':', 'LineWidth', 2, 'Color', 'r');
legend({'standard-repeat', 'standard-switch', 'bonus-repeat', 'bonus-switch'});
xline(0);
xline(800);
xlim([-500, 2000]);

% Create a plot of ERPs, averaged across subjects
figure;
plot(erp_times, grand_averages(5, :), '-', 'LineWidth', 2, 'Color', 'k');
hold on;
plot(erp_times, grand_averages(6, :), ':', 'LineWidth', 2, 'Color', 'k');
plot(erp_times, grand_averages(7, :), '-', 'LineWidth', 2, 'Color', 'r');
plot(erp_times, grand_averages(8, :), ':', 'LineWidth', 2, 'Color', 'r');
legend({'standard-start', 'standard-end', 'bonus-start', 'bonus-end'});
xline(0);
xline(800);
xlim([-500, 2000]);