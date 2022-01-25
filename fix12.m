clear all;

% PATH VARS
PATH_EEGLAB        = '/home/plkn/eeglab2021.1/';
PATH_LOGFILES      = '/mnt/data_dump/bocotilt/0_logfiles/';
PATH_RAW           = '/mnt/data_dump/bocotilt/0_eeg_raw/';

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Load force channels
EEG = pop_loadbv(PATH_RAW, ['VP12.vhdr'], [], [65, 66]);

% Prepare response channels
idx_left = find(strcmp({EEG.chanlocs.labels}, 'LeftKey'));
idx_right = find(strcmp({EEG.chanlocs.labels}, 'RightKey'));
resps_left = rescale(EEG.data(idx_left, :));
resps_right = rescale(EEG.data(idx_right, :));

% Iterate events
resp_lats = [];
counter = 0;
maxrt = 1500;
critval_responses = 0.5;
for e = 1 : length(EEG.event)

    % If trial
    if (strcmpi(EEG.event(e).type(1), {'S'}) & ismember(str2num(EEG.event(e).type(2 : 4)) - 40, [0 : 32]))

            % Get event code
            ecode = str2num(EEG.event(e).type(2 : 4)) - 40;

            % Lookup response
            left_rt = min(find(resps_left(EEG.event(e).latency + 800 : EEG.event(e).latency + 800 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);
            right_rt = min(find(resps_right(EEG.event(e).latency + 800 : EEG.event(e).latency + 800 + (maxrt / (1000 / EEG.srate))) >= critval_responses)) * (1000 / EEG.srate);

            if isempty(left_rt)
                left_rt = NaN;
            end
            if isempty(right_rt)
                right_rt = NaN;
            end

            % Break if suficient
            counter = counter + 1;
            resp_lats(counter, :) = [ecode, left_rt, right_rt];
            if counter >= 5
                break;
            end

    end
end