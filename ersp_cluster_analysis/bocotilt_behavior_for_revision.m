
% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_TF_DATA     = '/mnt/data_dump/bocotilt/4_ersp/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip-master/';
PATH_OUT         = '/mnt/data_dump/bocotilt/4_ersp/out/';
PATH_SELF_REPORT = '/mnt/data_dump/bocotilt/0_logfiles/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Loop subjects
behavior_rt = [];
behavior_ac = [];
for s = 1 : length(subject_list)

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

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
    % 24: Point(s) earned 

    % Exclude trials
    to_keep = EEG.trialinfo(:, 2) > 4 &...
    EEG.trialinfo(:, 23) > 1;
    EEG.trialinfo = EEG.trialinfo(to_keep, :);

    % Add to trialinfo
    for t = 1 : size(EEG.trialinfo, 1)

        % If correct and color task and rt < color-thresh
        if EEG.trialinfo(t, 18) == 1 & EEG.trialinfo(t, 5) == 0 & EEG.trialinfo(t, 15) <= EEG.trialinfo(t, 16)
            EEG.trialinfo(t, 24) = 1;

        % If correct and tilt task and rt < tilt-thresh
        elseif EEG.trialinfo(t, 18) == 1 & EEG.trialinfo(t, 5) == 1 & EEG.trialinfo(t, 15) <= EEG.trialinfo(t, 17)
            EEG.trialinfo(t, 24) = 1;

        % else...
        else
            EEG.trialinfo(t, 24) = 0;
        end
    end

    % Loop conditions
    counter = 0;
    for bon = 1 : 2
        for swi = 1 : 2
            for tsk = 1 : 2

                counter = counter + 1;

                % Get condition idx
                idx_condition = EEG.trialinfo(:, 4) == bon - 1 & EEG.trialinfo(:, 10) == swi - 1 & EEG.trialinfo(:, 5) == tsk - 1;

                % Get correct_idx for condition
                idx_correct = EEG.trialinfo(:, 18) == 1 & idx_condition;

                % Get accuracy
                ac = sum(idx_correct) / sum(idx_condition);

                % Get rt
                rt = mean(EEG.trialinfo(idx_correct, 15));

                % Write to matrices
                behavior_rt(s, counter) = rt;
                behavior_ac(s, counter) = ac; 

            end
        end
    end
end

% Perform rmANOVA for rt
varnames = {'id', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'};
t = table([1 : numel(subject_list)]', behavior_rt(:, 1), behavior_rt(:, 2), behavior_rt(:, 3), behavior_rt(:, 4), behavior_rt(:, 5), behavior_rt(:, 6), behavior_rt(:, 7), behavior_rt(:, 8), 'VariableNames', varnames);
within = table({'std'; 'std'; 'std'; 'std'; 'bon'; 'bon'; 'bon'; 'bon'}, {'rep'; 'rep'; 'swi'; 'swi'; 'rep'; 'rep'; 'swi'; 'swi'}, {'color'; 'tilt'; 'color'; 'tilt'; 'color'; 'tilt'; 'color'; 'tilt'}, 'VariableNames', {'bonus', 'switch', 'task'});
rm = fitrm(t, 'b1-b8~1', 'WithinDesign', within);
anova_rt = ranova(rm, 'WithinModel', 'bonus + switch + task + bonus*switch*task');
anova_rt

rt_means = mean(behavior_rt);
figure()
plot([1,2], rt_means([1, 5]), 'k','Linewidth', 2)
hold on
plot([1,2], rt_means([3, 7]), ':k','Linewidth', 2)
plot([1,2], rt_means([2, 6]), 'r','Linewidth', 2)
plot([1,2], rt_means([4, 8]), ':r','Linewidth', 2)
xticks([1, 2])
xlim([0.8, 2.2])
ylabel('ms')
xticklabels({'low reward', 'high reward'})
legend({'repeat - color', 'switch - color', 'repeat - orientation', 'switch - orientation'})


% Perform rmANOVA for accuracy

varnames = {'id', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'};
t = table([1 : numel(subject_list)]', behavior_ac(:, 1), behavior_ac(:, 2), behavior_ac(:, 3), behavior_ac(:, 4), behavior_ac(:, 5), behavior_ac(:, 6), behavior_ac(:, 7), behavior_ac(:, 8), 'VariableNames', varnames);
within = table({'std'; 'std'; 'std'; 'std'; 'bon'; 'bon'; 'bon'; 'bon'}, {'rep'; 'rep'; 'swi'; 'swi'; 'rep'; 'rep'; 'swi'; 'swi'}, {'color'; 'tilt'; 'color'; 'tilt'; 'color'; 'tilt'; 'color'; 'tilt'}, 'VariableNames', {'bonus', 'switch', 'task'});
rm = fitrm(t, 'b1-b8~1', 'WithinDesign', within);
anova_ac = ranova(rm, 'WithinModel', 'bonus + switch + task + bonus*switch*task');
anova_ac


