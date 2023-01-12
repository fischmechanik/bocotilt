% This removes all variables from the matlab workspace.
% At the beginning of a script this can help to prevent chaos...
clear all;

% Here ist the ERP data:
% https://ocfromthefuture.ifado.de:8081/owncloud/index.php/s/b0tt4Xa5yXxF15v
% Downoad and save to a folder.

% First, set up some path variables
path_erp_data = '/home/plkn/Desktop/erp_analysis_wouter/'; % (The path to the erp-data file)
path_eeglab   = '/home/plkn/eeglab/'; % (you need to download eeglab and put the path to the folder here)
path_results  = '/home/plkn/Desktop/erp_analysis_wouter/'; % Same as input path for now. But can be anything...

% Initialize eeglab 
addpath(path_eeglab);
eeglab;

% Load the erp data:
load([path_erp_data, 'erp_bocotilt.mat']);


% The '.mat' file contains 3 variables:
%
% [1] The first one is 'erp_matrix', a 5-dimensional matrix with the dimensions:
% 1 = subjects (N=26) 
% 2 = bonus condition (N=2, 1 is low reward, 2 is high reward)
% 3 = task switching (N=2, 1 is repetition, 2 is switch)
% 4 = electrodes (N=127)
% 5 = time points (N=451, ranging from - 200 ms to 1600 ms relative to the onset of the task-cue)
%
% [2] The second variable is 'eeg_times', a vector containing the 451 ms-values belonging to the time dimension.
%
% [3] The third is 'chanlocs', a data structure describing the 127 electrode positions. You can
% double click this structure to inspect which electrode number belongs to which electrode
% label/position. You can also use this data-structure to pass it to an eeglab function for
% plotting topographies.

% Ok, a quick example of how to index matrices, using the erp-data.
% Lets assume we want to look at POz since we believe its an awsome electrode, 'chanlocs' says POz is at position 63.
% We also want to specifically look at the 5th participant.
% We want to compare repeat and switch erp in the high reward condition.
% Note that the 'squeeze()' function removes all dimensions that are not needed, that is that are of one element length.
erp_repeat = squeeze(erp_matrix(5, 2, 1, 63, :));
erp_switch = squeeze(erp_matrix(5, 2, 2, 63, :));

% As an example, lets plot these ERPs of participant 5 from POz. High reward repeat and high reward switch:
figure()
plot(eeg_times, erp_repeat, 'Linewidth', 1.5)
hold on
plot(eeg_times, erp_switch, 'Linewidth', 1.5)
xlabel('ms')
ylabel('mV^2')
legend({'repeat', 'switch'})
title('The 5th participant has ERPs')

% Lets plot the same condition comparison, but this time the average across all participants
% Note that we use the mean function to average specifically across the first dimension here.
erp_repeat = squeeze(mean(erp_matrix(:, 2, 1, 63, :), 1));
erp_switch = squeeze(mean(erp_matrix(:, 2, 2, 63, :), 1));

% As an example, lets plot these ERPs of participant 5 from POz. High reward repeat and high reward switch:
figure()
plot(eeg_times, erp_repeat, 'Linewidth', 1.5)
hold on
plot(eeg_times, erp_switch, 'Linewidth', 1.5)
xlabel('ms')
ylabel('mV^2')
legend({'repeat', 'switch'})
title('All participants has ERPs')

% In this example, we see that there is a difference in the cue-P3. At 600-700 ms this difference seems large. 
% Lets plot a topography at this time point.

% First, lets identify the time-indices belonging to this time window:
time_idx = eeg_times >= 600 & eeg_times <= 700;

% Now we can use these indices to index the temporal dimension and average the values within this time window.
% 'squeeze()' the again gets rid of the now useless temporal dimension:
erp_matrix_timewin = squeeze(mean(erp_matrix(:, :, :, :, time_idx), 5));

% Again, select repeat and switch in the high reward condition and average scross participants.
% The resulting data is avector of length 127. One value for each electrode.
topo_values_repeat = squeeze(mean(erp_matrix_timewin(:, 2, 1, :), 1));
topo_values_switch = squeeze(mean(erp_matrix_timewin(:, 2, 2, :), 1));

% Lets use eeglabs topoplot function to plot the mV values of this timewindow for both conditions:
figure()
subplot(1, 2, 1)
topoplot(topo_values_repeat, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
title('repeat in high reward - 600-700 ms')
colorbar()
clim([-3, 3])
subplot(1, 2, 2)
topoplot(topo_values_switch, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
title('switch in high reward - 600-700 ms')
colorbar()
clim([-3, 3])

% Just as an example, lets use a for loop to plot topographies over time. We use 100 ms wide time windows
% starting at 0 ms.
figure()
for t = 1 : 12

    % Get time window index
    time_idx = eeg_times >= t * 100 - 100 & eeg_times <= t * 100;
    erp_matrix_timewin = squeeze(mean(erp_matrix(:, :, :, :, time_idx), 5));

    % Calculate topo values
    topo_values_repeat = squeeze(mean(erp_matrix_timewin(:, 2, 1, :), 1));
    topo_values_switch = squeeze(mean(erp_matrix_timewin(:, 2, 2, :), 1));

    % Plot that
    subplot(2, 12, t)
    topoplot(topo_values_repeat, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    clim([-3, 3])
    subplot(2, 12, t + 12)
    topoplot(topo_values_switch, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
    clim([-3, 3])

end

% Lets define frontal and posterior electrode patches that we want to use for analyses
frontal_idx = [33, 17, 34, 65, 66, 21, 127, 22, 97, 98, 35, 18, 36];
posterior_idx = [37, 19, 38, 71, 72, 45, 63, 46, 107, 108];

% Plot these electrode patches to show which electrodes are included
figure()
subplot(1, 2, 1)
topoplot(ones(1, 127), chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {frontal_idx, '.', 'k', 14, 1});
colormap('white')
title('frontal electrode patch')
subplot(1, 2, 2)
topoplot(ones(1, 127), chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {posterior_idx, '.', 'k', 14, 1});
colormap('white')
title('posterior electrode patch')

% Ok, now we can plot the ERP in time (the time-course) and in space (the topography).
% Next is an example how to parameterize a component.
% Lets stick to this cue dependent P3 we saw in the example plots above. We want to use a time
% window from 500 to 700 ms and the posterior electrode patch. We want to have a single value
% for each participant in each condition. We can then use this value in an ANOVA.

% Reset out time indices (larger time window now...)
time_idx = eeg_times >= 500 & eeg_times <= 700;

% We use loops to iterate participants and conditions

% We want to store our results here (initialize as an empty matrix)
anova_table = [];

% Loop participants
for s = 1 : size(erp_matrix, 1) % First dimension has length of number of participants...

    % A counter
    counter = 0;

    % Loop high versus low reward
    for rew = 1 : 2

        % Loop repeat / switch
        for sw = 1 : 2

            % Since we have a specific subject-condition combination here, we
            % can index the matrix accordingly. In time and space we are not specific
            % yet, as we have a patch of electrodes and multiple timepoints constitute
            % The time window. So we average across these dimensions. The result is a
            % single value.
            erp_value = squeeze(mean(erp_matrix(s, rew, sw, posterior_idx, time_idx), [4, 5]));

            % Increase counter
            counter = counter + 1;

            % Now we store this value in the 'anova_table'. We need the counter for this to
            % find the correct column.
            anova_table(s, counter) = erp_value;

        end
    end
end

% You can now save this table as a csv file to use in R or SPSS or whatever...
writematrix(anova_table, [path_results, 'anova_table.csv']);

% As an alternative, Matlab has also an ANOVA function. It is a bit clunky, but
% if you know how to set it up it works just fine. 
% An example:
varnames = {'id', 'cond1', 'cond2', 'cond3', 'cond4'};
t = table([1 : size(erp_matrix, 1)]', anova_table(:, 1), anova_table(:, 2), anova_table(:, 3), anova_table(:, 4), 'VariableNames', varnames);
within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
rm = fitrm(t, 'cond1-cond4~1', 'WithinDesign', within);
anova_posterior_p3 = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');

% Print anova results to matlab console
anova_posterior_p3

% Lets get the posterior ERPs again, for all conditions
erp_posterior_low_reward_repeat  = squeeze(mean(erp_matrix(:, 1, 1, posterior_idx, :), [1, 4]));
erp_posterior_low_reward_switch  = squeeze(mean(erp_matrix(:, 1, 2, posterior_idx, :), [1, 4]));
erp_posterior_high_reward_repeat = squeeze(mean(erp_matrix(:, 2, 1, posterior_idx, :), [1, 4]));
erp_posterior_high_reward_switch = squeeze(mean(erp_matrix(:, 2, 2, posterior_idx, :), [1, 4]));

% Lets plot all conditions in a single plot
figure()
plot(eeg_times, erp_posterior_low_reward_repeat, 'k-', 'LineWidth', 2)
hold on;
plot(eeg_times, erp_posterior_low_reward_switch, 'k:', 'LineWidth', 2)
plot(eeg_times, erp_posterior_high_reward_repeat, 'r-', 'LineWidth', 2)
plot(eeg_times, erp_posterior_high_reward_switch, 'r:', 'LineWidth', 2)
title('posterior electrodes ERP')
legend({'low-repeat', 'low-switch', 'high-repeat', 'high-switch', 'target-onset'})
xline([0, 800])