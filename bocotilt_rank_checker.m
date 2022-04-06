% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_ICSET       = '/mnt/data_dump/bocotilt/1_icset/';

% Subject list
subject_list = {'VP08', 'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP24'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;


% Loop subjects
data_ranks = [];
for s = 1 : length(subject_list)

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_icset.set'], 'filepath', PATH_ICSET, 'loadmode', 'all');
    data_ranks(s, 1) = size(EEG.icawinv, 2);

end
