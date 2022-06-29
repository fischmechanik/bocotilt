% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.0/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_GED         = '/mnt/data_dump/bocotilt/7_ged/';
PATH_VEUSZ       = '/mnt/data_dump/bocotilt/veusz/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31'};


%subject_list = {'VP09', 'VP17', 'VP25', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part1'};

% Part 1: Calculate ged
if ismember('part1', to_execute)

    std = [];
    bon = [];

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

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
        % 11: prev_switch
        % 12: prev_accuracy
        % 13: correct_response
        % 14: response_side
        % 15: rt
        % 16: rt_thresh_color
        % 17: rt_thresh_tilt
        % 18: accuracy
        % 19: position_color
        % 20: position_tilt
        % 21: position_target
        % 22: position_distractor    
        % 23: sequence_position
        
        % Exclude trials
        to_keep = EEG.trialinfo(:, 2) > 4 & EEG.trialinfo(:, 18) == 1 & EEG.trialinfo(:, 10) ~= -1 & EEG.trialinfo(:, 11) ~= -1;
        eeg_data = eeg_data(:, :, to_keep);
        EEG.trialinfo = EEG.trialinfo(to_keep, :);
        EEG.trials = sum(to_keep);

        % Apply surface laplacian
        X = [EEG.chanlocs(:).X]';
        Y = [EEG.chanlocs(:).Y]';
        Z = [EEG.chanlocs(:).Z]';
        eeg_data = laplacian_perrinX(eeg_data, X, Y, Z, 12);

        % Construct filter
        nyquist = EEG.srate / 2;
        h_pass = 8;
        l_pass = 12;
        transition_width = 0.2;
        filter_order = round(3 * (EEG.srate / h_pass));
        filter_freqs = [0, (1 - transition_width) * h_pass, h_pass, l_pass, (1 + transition_width) * l_pass, nyquist] / nyquist; 
        filter_response = [0, 0, 1, 1, 0, 0];
        filter_weights = firls(filter_order, filter_freqs, filter_response);

        % Reshape to 2d
        eeg_data_2d = reshape(eeg_data, [EEG.nbchan, EEG.pnts * EEG.trials]);

        % Select channels
        chan1 = 3;
        chan2 = 4;

        % Apply filter
        chan1_filtered_1d = filtfilt(filter_weights, 1, eeg_data_2d(chan1, :));
        chan2_filtered_1d = filtfilt(filter_weights, 1, eeg_data_2d(chan2, :));

        % Get power
        chan1_power_1d = abs(hilbert(chan1_filtered_1d')) .^ 2;
        chan2_power_1d = abs(hilbert(chan2_filtered_1d')) .^ 2;

        chan1_filtered2_1d = real(hilbert(chan1_filtered_1d'));
        chan2_filtered2_1d = real(hilbert(chan2_filtered_1d'));

        % Back to 2d
        chan1_eeg_2d = reshape(eeg_data_2d(chan1, :), [EEG.pnts, EEG.trials]);
        chan2_eeg_2d = reshape(eeg_data_2d(chan2, :), [EEG.pnts, EEG.trials]);
        chan1_filtered_2d = reshape(chan1_filtered_1d, [EEG.pnts, EEG.trials]);
        chan2_filtered_2d = reshape(chan2_filtered_1d, [EEG.pnts, EEG.trials]);
        chan1_power_2d = reshape(chan1_power_1d, [EEG.pnts, EEG.trials]);
        chan2_power_2d = reshape(chan2_power_1d, [EEG.pnts, EEG.trials]);

        % Prune
        prune_times = [-500, 1800];
        chan1_eeg_2d =           chan1_eeg_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        chan2_eeg_2d =           chan2_eeg_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        chan1_filtered_2d = chan1_filtered_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        chan2_filtered_2d = chan2_filtered_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        chan1_power_2d =       chan1_power_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        chan2_power_2d =       chan2_power_2d(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        prune_time = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

        % Calculate left - right
        diff_power_2d = chan1_power_2d - chan2_power_2d;

        % Get difference for standard and bonus condition
        std(s, :) = mean(diff_power_2d(:, EEG.trialinfo(:, 4) == 0), 2);
        bon(s, :) = mean(diff_power_2d(:, EEG.trialinfo(:, 4) == 1), 2);

    end

    figure()
    plot(prune_time, [mean(std, 1); mean(bon, 1)])



end % End part1