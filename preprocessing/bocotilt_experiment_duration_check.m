clear all;

% PATH VARS
PATH_EEGLAB        = '/home/plkn/eeglab2022.1/';
PATH_RAW           = '/mnt/data_dump/bocotilt/0_eeg_raw/';

% Subjects
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

durations = [];

% Iterate subjects
for s = 1 : length(subject_list)

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load
    EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);

    % NaN init
    lat_start = NaN;
    lat_end = NaN;

    % Iterate for start/end latencies
    for e = 1 : length(EEG.event)
        if strcmpi(EEG.event(e).type, 'S254')
            lat_start = EEG.event(e).latency;
        end
        if strcmpi(EEG.event(e).type, 'S255')
            lat_end = EEG.event(e).latency;
        end
    end

    if isnan(lat_start)
        lat_start = 0;
    end


    durations(s, :) = [id, lat_start, lat_end, lat_end - lat_start];

end

average_duration = mean(durations(:, 4) ./ 1000)

std(durations(:, 4) ./ 1000)