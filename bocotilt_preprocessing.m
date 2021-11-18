clear all;

% PATH VARS
PATH_EEGLAB        = '/home/plkn/eeglab2021.1/';
PATH_LOGFILES      = '/mnt/data_dump/bocotilt/0_logfiles/';
PATH_RAW           = '/mnt/data_dump/bocotilt/0_eeg_raw/';
PATH_ICSET         = '/mnt/data_dump/bocotilt/1_icset/';
PATH_AUTOCLEANED   = '/mnt/data_dump/bocotilt/2_autocleaned/';

% Subjects
subject_list = {'VP06', 'VP07'};

% Test switch                  
if true
    subject_list = {'VP08'};
end

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;
channel_location_file = which('dipplot.m');
channel_location_file = channel_location_file(1 : end - length('dipplot.m'));
channel_location_file = [channel_location_file, 'standard_BESA/standard-10-5-cap385.elp'];

% Iterate subjects
for s = 1 : length(subject_list)

    % participant identifiers
    subject = subject_list{s};
    id = str2num(subject(3 : 4));

    % Load
    if  id == 4
        EEG1 = pop_loadbv(PATH_RAW, [subject, '_01.vhdr'], [], []);
        EEG2 = pop_loadbv(PATH_RAW, [subject, '_02.vhdr'], [], []);
        EEG = pop_mergeset(EEG1, EEG2);
    else
        EEG = pop_loadbv(PATH_RAW, [subject, '.vhdr'], [], []);
    end

    % Fork response button channels
    RESPS = pop_select(EEG, 'channel', [65, 66]);
    EEG = pop_select(EEG, 'nochannel', [65, 66]);

    % Open log file
    fid = fopen([PATH_LOGFILES, subject, '_degreeLog.txt'], 'r');

    % Extract lines as strings
    logcell = {};
    tline = fgetl(fid);
    while ischar(tline)
        logcell{end + 1} = tline;
        tline = fgetl(fid);
    end

    % Delete header
    logcell(1 : 3) = [];

    % Get color and tilt positions in probe display (numbers 1-8)
    positions = [];
    for l = 1 : length(logcell)
        line_values = split(logcell{l}, ' ');
        positions(l, 1) = str2num(line_values{8});
        positions(l, 2) = str2num(line_values{10});
    end

    % Event coding
    EEG = bocotilt_event_coding(EEG, RESPS, positions);

    % Add channel locations
    EEG = pop_chanedit(EEG, 'lookup', channel_location_file);
    EEG.chanlocs_original = EEG.chanlocs;

    % Remove data at boundaries
    EEG = pop_rmdat(EEG, {'boundary'}, [0, 1], 1);

    % Resample data
    EEG = pop_resample(EEG, 200);

    % Filter
    EEG = pop_basicfilter(EEG, [1 : EEG.nbchan], 'Cutoff', [1, 30], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 4, 'RemoveDC', 'on', 'Boundary', 'boundary');  
        
    % Bad channel detection
    [EEG, i1] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 10, 'norm', 'on', 'measure', 'kurt');
    [EEG, i2] = pop_rejchan(EEG, 'elec', [1 : EEG.nbchan], 'threshold', 5, 'norm', 'on', 'measure', 'prob');
    EEG.chans_rejected = [i1, i2];

    % Reref common average
    EEG = pop_reref(EEG, []);

    % Determine rank of data
    dataRank = sum(eig(cov(double(EEG.data'))) > 1e-6); 

    % Interpolate channels
    EEG = pop_interp(EEG, EEG.chanlocs_original, 'spherical');

    % Epoch data
    EEG = pop_epoch(EEG, {'trial'}, [-0.8, 2.6], 'newname', [subject '_epoched'], 'epochinfo', 'yes');
    EEG = pop_rmbase(EEG, [-200, 0]);

    % Autoreject trials
    [EEG, rejsegs] = pop_autorej(EEG, 'nogui', 'on', 'threshold', 1000, 'startprob', 5, 'maxrej', 5);
    EEG.n_segs_rejected = length(rejsegs);

    % Fin standard latency of event in epoch
    lats = [];
    for e = 1 : length(EEG.event)
        lats(end+1) = mod(EEG.event(e).latency, EEG.pnts);
    end
    lat_mode = mode(lats);
    
    % Compile a trialinfo matrix
    trialinfo = [];
    counter = 0;
    for e = 1 : length(EEG.event)
        if strcmpi(EEG.event(e).type, 'trial') & (mod(EEG.event(e).latency, EEG.pnts) == lat_mode)

            counter = counter + 1;

            % Compile table
            trialinfo(counter, :) = [id,...
                                        EEG.event(e).block_nr,...
                                        EEG.event(e).trial_nr,...
                                        EEG.event(e).bonustrial,...
                                        EEG.event(e).tilt_task,...
                                        EEG.event(e).cue_ax,...
                                        EEG.event(e).target_red_left,...
                                        EEG.event(e).distractor_red_left,...
                                        EEG.event(e).response_interference,...
                                        EEG.event(e).task_switch,...
                                        EEG.event(e).correct_response,...
                                        EEG.event(e).response_side,...
                                        EEG.event(e).rt,...
                                        EEG.event(e).accuracy,...
                                        EEG.event(e).position_color,...
                                        EEG.event(e).position_tilt,...
                                        EEG.event(e).position_target,...
                                        EEG.event(e).position_distractor,...    
                                        EEG.event(e).sequence_position,...
                                        EEG.event(e).sequence_length,...
                                        ];

        end
    end

    % Save trialinfo
    EEG.trialinfo = trialinfo;
    writematrix(trialinfo, [PATH_AUTOCLEANED, subject, '_trialinfo.csv']);

    % Runica & ICLabel
    EEG = pop_runica(EEG, 'extended', 1, 'interrupt', 'on', 'PCA', dataRank);
    EEG = iclabel(EEG);

    % Find nobrainer
    EEG.nobrainer = find(EEG.etc.ic_classification.ICLabel.classifications(:, 1) < 0.7);

    % Save IC set
    pop_saveset(EEG, 'filename', [subject, '_icset.set'], 'filepath', PATH_ICSET, 'check', 'on', 'savemode', 'twofiles');

    % Remove components
    EEG = pop_subcomp(EEG, EEG.nobrainer, 0);

    % Save clean data
    pop_saveset(EEG, 'filename', [subject, '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'check', 'on', 'savemode', 'twofiles');
    
end