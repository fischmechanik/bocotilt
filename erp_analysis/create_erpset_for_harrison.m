% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_OUT         = '/mnt/data_dump/bocotilt/data_harrison/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Trialinfo columns
%  1: id
%  2: block_nr
%  3: trial_nr XXXXXXXXXXXXXXXXXXXXXXXXXX
%  4: bonustrial XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
% 23: sequence_position  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

% Loop subjects
for s = 1 : length(subject_list)

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Remove bonus info from event struct
    EEG.event = rmfield(EEG.event, 'bonustrial');
    EEG.event = rmfield(EEG.event, 'sequence_position');
    EEG.event = rmfield(EEG.event, 'trial_nr');

    for e = 1 : length(EEG.event)
        EEG.event(e).urevent = 0;
    end

    % Remove from trialinfo
    EEG.trialinfo(:, [3, 4, 23]) = [];

    % New trialinfo columns
    %  1: id
    %  2: block_nr
    %  3: tilt_task
    %  4: cue_ax
    %  5: target_red_left
    %  6: distractor_red_left
    %  7: response_interference
    %  8: task_switch
    %  9: prev_switch
    % 10: prev_accuracy
    % 11: correct_response
    % 12: response_side
    % 13: rt
    % 14: rt_thresh_color
    % 15: rt_thresh_tilt
    % 16: accuracy
    % 17: position_color
    % 18: position_tilt
    % 19: position_target
    % 20: position_distractor    

    pop_saveset(EEG, 'filename', [subject_list{s}, '_erp.set'], 'filepath', PATH_OUT, 'check', 'on');


end % End subject loop








