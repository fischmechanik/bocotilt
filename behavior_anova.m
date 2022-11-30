
% Path to behavioral data
path_behavioral_data = '/home/plkn/repos/bocotilt/';

% Load data
load([path_behavioral_data, 'behavioral_data.mat']);

% Determine number of participants
n_subjects = size(behavior_rt, 1);

% Perform rmANOVA for rt
varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
t = table([1 : n_subjects]', behavior_rt(:, 1), behavior_rt(:, 2), behavior_rt(:, 3), behavior_rt(:, 4), 'VariableNames', varnames);
within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
anova_rt = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
anova_rt

% Perform rmANOVA for accuracy
varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
t = table([1 : n_subjects]', behavior_ac(:, 1), behavior_ac(:, 2), behavior_ac(:, 3), behavior_ac(:, 4), 'VariableNames', varnames);
within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
anova_ac = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
anova_ac