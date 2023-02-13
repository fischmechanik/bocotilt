% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% We need this...
res_rt = [];
res_acc = [];

% Trialinfo columns
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

% Loop subjects
for s = 1 : length(subject_list)

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Exclude trials
    to_keep = EEG.trialinfo(:, 2) > 4 & EEG.trialinfo(:, 23) > 1;
    EEG.trialinfo = EEG.trialinfo(to_keep, :);

    % Loop 3 factors
    counter = 0;
    for rew = 0 : 1
        for swi = 0 : 1
            for tilt = 0 : 1

                % Count!!!!!!
                counter = counter + 1;

                % Get condition idx
                condition_idx = EEG.trialinfo(:, 4) == rew & EEG.trialinfo(:, 10) == swi & EEG.trialinfo(:, 5) == tilt;

                % Get correct trials idx
                correct_idx = EEG.trialinfo(:, 18) == 1;

                % Get accuracy
                acc = sum(correct_idx & condition_idx) / sum(condition_idx);

                % Get RT
                rt = sum(EEG.trialinfo(condition_idx & correct_idx, 15))  / sum(condition_idx & correct_idx);

                % Fill result matrices
                res_rt(s, counter) = rt;
                res_acc(s, counter) = acc;

            end
        end
    end


end % End subject loop

% ANOVA rt
varnames = {'id', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'};
t = table([1 : length(subject_list)]', res_rt(:, 1), res_rt(:, 2), res_rt(:, 3), res_rt(:, 4), res_rt(:, 5), res_rt(:, 6), res_rt(:, 7), res_rt(:, 8), 'VariableNames', varnames);
within = table({'std'; 'std'; 'std'; 'std'; 'bon'; 'bon'; 'bon'; 'bon'},...
               {'rep'; 'rep'; 'swi'; 'swi'; 'rep'; 'rep'; 'swi'; 'swi'},...
               {'col'; 'til'; 'col'; 'til'; 'col'; 'til'; 'col'; 'til'},...
                'VariableNames', {'bonus', 'switch', 'task'});
                rm = fitrm(t, 'b1-b8~1', 'WithinDesign', within);
                anova_rt = ranova(rm, 'WithinModel', 'bonus + switch + task + bonus*switch + bonus*task + switch*task + bonus*switch*task');
                anova_rt

% ANOVA accuracy
varnames = {'id', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'};
t = table([1 : length(subject_list)]', res_acc(:, 1), res_acc(:, 2), res_acc(:, 3), res_acc(:, 4), res_acc(:, 5), res_acc(:, 6), res_acc(:, 7), res_acc(:, 8), 'VariableNames', varnames);
within = table({'std'; 'std'; 'std'; 'std'; 'bon'; 'bon'; 'bon'; 'bon'},...
               {'rep'; 'rep'; 'swi'; 'swi'; 'rep'; 'rep'; 'swi'; 'swi'},...
               {'col'; 'til'; 'col'; 'til'; 'col'; 'til'; 'col'; 'til'},...
                'VariableNames', {'bonus', 'switch', 'task'});
                rm = fitrm(t, 'b1-b8~1', 'WithinDesign', within);
                anova_acc = ranova(rm, 'WithinModel', 'bonus + switch + task + bonus*switch + bonus*task + switch*task + bonus*switch*task');
                anova_acc











