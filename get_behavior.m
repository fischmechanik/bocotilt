%clear all;

% PATH VARS
PATH_EEGLAB        = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED   = '/home/plkn/bocotilt/cleaned/';
PATH_OUT           = '/home/plkn/bocotilt/behavior/'
;
% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Preprostats matrix
prepro_stats = [];

% Collector
data = [];

% Iterate subjects
for s = 1 : length(subject_list)

    %  Load info
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');
 
    % Trialinfo columns:
    % 01: id
    % 02: block_nr
    % 03: trial_nr
    % 04: bonustrial
    % 05: tilt_task
    % 06: cue_ax
    % 07: target_red_left
    % 08: distractor_red_left
    % 09: response_interference
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
    % 23: sequence_pos       
    
    % Exclude non-responses and first-in-sequence trials (no switch nor repeat...)
    to_keep = EEG.trialinfo(:, 18) ~= 2 & EEG.trialinfo(:, 18) ~= -1;

    % Get trials
    if s == 1
        data = EEG.trialinfo(to_keep, :);
    else
        data = vertcat(data, EEG.trialinfo(to_keep, :));
    end

end

% Save as csv
writematrix(data, [PATH_OUT, 'behavioral_data.csv']);





