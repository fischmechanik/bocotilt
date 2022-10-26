% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip-master/';
PATH_OUTPUT      = '/mnt/data_dump/bocotilt/99_erp_results/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init ft
addpath(PATH_FIELDTRIP);
ft_defaults;

% This is where we collect the ERPs
erp_matrix = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Trial data
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
    
    % The data matrix: channels x times x trials
    eeg_data = double(EEG.data);

    % Prune time
    idx_keep = EEG.times >= -200 & EEG.times <= 1600;
    eeg_times = EEG.times(idx_keep);
    eeg_data = eeg_data(:, idx_keep, :);

    % Exclude trials
    to_keep = EEG.trialinfo(:, 2) > 4 &...
              EEG.trialinfo(:, 23) > 1;
    eeg_data = double(eeg_data(:, :, to_keep));
    EEG.trialinfo = EEG.trialinfo(to_keep, :);

    % Get condition idx
    idx_std_rep = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 0;
    idx_std_swi = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 1;
    idx_bon_rep = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 0;
    idx_bon_swi = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 1;

    % Collect erps
    erp_matrix(s, 1, 1, :, :) = squeeze(mean(eeg_data(:, :, idx_std_rep), 3));
    erp_matrix(s, 1, 2, :, :) = squeeze(mean(eeg_data(:, :, idx_std_swi), 3));
    erp_matrix(s, 2, 1, :, :) = squeeze(mean(eeg_data(:, :, idx_bon_rep), 3));
    erp_matrix(s, 2, 2, :, :) = squeeze(mean(eeg_data(:, :, idx_bon_swi), 3));

end

% Seperate cue and target erps
idx_cue = eeg_times >= -200 & eeg_times <= 800;
idx_target = eeg_times >= 600 & eeg_times <= 1600;
erp_cue = erp_matrix(:, :, :, :, idx_cue);
erp_target = erp_matrix(:, :, :, :, idx_target);
cue_times = eeg_times(idx_cue);
target_times = eeg_times(idx_target) - 800;

% Apply baseline to target erps
for s = 1 : length(subject_list)
    for rew = 1 : 2
        for sw = 1 : 2
            for ch = 1 : EEG.nbchan
                tmp = squeeze(erp_target(s, rew, sw, ch, :));
                bl = mean(tmp(target_times >= -200 & target_times <= 0));
                erp_target(s, rew, sw, ch, :) = tmp - bl;
            end
        end
    end
end

% Get frontal erps
frontal_idx = [33, 17, 34, 65, 66, 21, 127, 22, 97, 98, 35, 18, 36];
erp_all_frontal = squeeze(mean(erp_matrix(:, :, :, frontal_idx, :), 4));
erp_cue_frontal = squeeze(mean(erp_cue(:, :, :, frontal_idx, :), 4));
erp_target_frontal = squeeze(mean(erp_target(:, :, :, frontal_idx, :), 4));

% Get posterior erps
posterior_idx = [37, 19, 38, 71, 72, 45, 63, 46, 107, 108];
erp_all_posterior = squeeze(mean(erp_matrix(:, :, :, posterior_idx, :), 4));
erp_cue_posterior = squeeze(mean(erp_cue(:, :, :, posterior_idx, :), 4));
erp_target_posterior = squeeze(mean(erp_target(:, :, :, posterior_idx, :), 4));

% Get grand averages for plotting
ga_cue_frontal_std_rep = mean(squeeze(erp_cue_frontal(:, 1, 1, :)), 1);
ga_cue_frontal_std_swi = mean(squeeze(erp_cue_frontal(:, 1, 2, :)), 1);
ga_cue_frontal_bon_rep = mean(squeeze(erp_cue_frontal(:, 2, 1, :)), 1);
ga_cue_frontal_bon_swi = mean(squeeze(erp_cue_frontal(:, 2, 2, :)), 1);

ga_target_frontal_std_rep = mean(squeeze(erp_target_frontal(:, 1, 1, :)), 1);
ga_target_frontal_std_swi = mean(squeeze(erp_target_frontal(:, 1, 2, :)), 1);
ga_target_frontal_bon_rep = mean(squeeze(erp_target_frontal(:, 2, 1, :)), 1);
ga_target_frontal_bon_swi = mean(squeeze(erp_target_frontal(:, 2, 2, :)), 1);

ga_cue_posterior_std_rep = mean(squeeze(erp_cue_posterior(:, 1, 1, :)), 1);
ga_cue_posterior_std_swi = mean(squeeze(erp_cue_posterior(:, 1, 2, :)), 1);
ga_cue_posterior_bon_rep = mean(squeeze(erp_cue_posterior(:, 2, 1, :)), 1);
ga_cue_posterior_bon_swi = mean(squeeze(erp_cue_posterior(:, 2, 2, :)), 1);

ga_target_posterior_std_rep = mean(squeeze(erp_target_posterior(:, 1, 1, :)), 1);
ga_target_posterior_std_swi = mean(squeeze(erp_target_posterior(:, 1, 2, :)), 1);
ga_target_posterior_bon_rep = mean(squeeze(erp_target_posterior(:, 2, 1, :)), 1);
ga_target_posterior_bon_swi = mean(squeeze(erp_target_posterior(:, 2, 2, :)), 1);

ga_all_frontal_std_rep = mean(squeeze(erp_all_frontal(:, 1, 1, :)), 1);
ga_all_frontal_std_swi = mean(squeeze(erp_all_frontal(:, 1, 2, :)), 1);
ga_all_frontal_bon_rep = mean(squeeze(erp_all_frontal(:, 2, 1, :)), 1);
ga_all_frontal_bon_swi = mean(squeeze(erp_all_frontal(:, 2, 2, :)), 1);

ga_all_posterior_std_rep = mean(squeeze(erp_all_posterior(:, 1, 1, :)), 1);
ga_all_posterior_std_swi = mean(squeeze(erp_all_posterior(:, 1, 2, :)), 1);
ga_all_posterior_bon_rep = mean(squeeze(erp_all_posterior(:, 2, 1, :)), 1);
ga_all_posterior_bon_swi = mean(squeeze(erp_all_posterior(:, 2, 2, :)), 1);


figure()

subplot(2, 2, 1)
plot(cue_times, ga_cue_frontal_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(cue_times, ga_cue_frontal_std_swi, 'k:', 'LineWidth', 2)
plot(cue_times, ga_cue_frontal_bon_rep, 'r-', 'LineWidth', 2)
plot(cue_times, ga_cue_frontal_bon_swi, 'r:', 'LineWidth', 2)
title('cue - frontal')
legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi', 'stim-onset'})
xline([0])

subplot(2, 2, 2)
plot(target_times, ga_target_frontal_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(target_times, ga_target_frontal_std_swi, 'k:', 'LineWidth', 2)
plot(target_times, ga_target_frontal_bon_rep, 'r-', 'LineWidth', 2)
plot(target_times, ga_target_frontal_bon_swi, 'r:', 'LineWidth', 2)
title('target - frontal')
xline([0])

subplot(2, 2, 3)
plot(cue_times, ga_cue_posterior_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(cue_times, ga_cue_posterior_std_swi, 'k:', 'LineWidth', 2)
plot(cue_times, ga_cue_posterior_bon_rep, 'r-', 'LineWidth', 2)
plot(cue_times, ga_cue_posterior_bon_swi, 'r:', 'LineWidth', 2)
title('cue - posterior')
xline([0])

subplot(2, 2, 4)
plot(target_times, ga_target_posterior_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(target_times, ga_target_posterior_std_swi, 'k:', 'LineWidth', 2)
plot(target_times, ga_target_posterior_bon_rep, 'r-', 'LineWidth', 2)
plot(target_times, ga_target_posterior_bon_swi, 'r:', 'LineWidth', 2)
title('target - posterior')
xline([0])




figure()

subplot(2, 1, 1)
plot(eeg_times, ga_all_frontal_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(eeg_times, ga_all_frontal_std_swi, 'k:', 'LineWidth', 2)
plot(eeg_times, ga_all_frontal_bon_rep, 'r-', 'LineWidth', 2)
plot(eeg_times, ga_all_frontal_bon_swi, 'r:', 'LineWidth', 2)
title('frontal')
legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi', 'stim-onset'})
xline([0, 800])

subplot(2, 1, 2)
plot(eeg_times, ga_all_posterior_std_rep, 'k-', 'LineWidth', 2)
hold on;
plot(eeg_times, ga_all_posterior_std_swi, 'k:', 'LineWidth', 2)
plot(eeg_times, ga_all_posterior_bon_rep, 'r-', 'LineWidth', 2)
plot(eeg_times, ga_all_posterior_bon_swi, 'r:', 'LineWidth', 2)
title('posterior')
xline([0, 800])





