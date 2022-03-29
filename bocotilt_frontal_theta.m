% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_TFDECOMP    = '/mnt/data_dump/bocotilt/4_frontal_theta/';

% Subject list
subject_list = {'VP08', 'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP24'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part2'};

% Part 1: tf-decomp
if ismember('part1', to_execute)

    % Set complex Morlet wavelet parameters
    n_frq = 25;
    frqrange = [2, 25];
    tfres_range = [400, 100];

    % Load data info from file
    EEG = pop_loadset('filename', [subject_list{1} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Set wavelet time
    wtime = -2 : 1 / EEG.srate : 2;

    % Determine fft frqs
    hz = linspace(0, EEG.srate, length(wtime));

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = linspace(frqrange(1), frqrange(2), n_frq);
    fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

    % Init matrices for wavelets
    cmw = zeros(length(tf_freqs), length(wtime));
    cmwX = zeros(length(tf_freqs), length(wtime));
    tlim = zeros(1, length(tf_freqs));

    % These will contain the wavelet widths as full width at 
    % half maximum in the temporal and spectral domain
    obs_fwhmT = zeros(1, length(tf_freqs));
    obs_fwhmF = zeros(1, length(tf_freqs));

    % Create the wavelets
    for frq = 1 : length(tf_freqs)

        % Create wavelet with tapering gaussian corresponding to desired width in temporal domain
        cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);

        % Normalize wavelet
        cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));

        % Create normalized freq domain wavelet
        cmwX(frq, :) = fft(cmw(frq, :)) ./ max(fft(cmw(frq, :)));

        % Determine observed fwhmT
        midt = dsearchn(wtime', 0);
        cmw_amp = abs(cmw(frq, :)) ./ max(abs(cmw(frq, :))); % Normalize cmw amplitude
        obs_fwhmT(frq) = wtime(midt - 1 + dsearchn(cmw_amp(midt : end)', 0.5)) - wtime(dsearchn(cmw_amp(1 : midt)', 0.5));

        % Determine observed fwhmF
        idx = dsearchn(hz', tf_freqs(frq));
        cmwx_amp = abs(cmwX(frq, :)); 
        obs_fwhmF(frq) = hz(idx - 1 + dsearchn(cmwx_amp(idx : end)', 0.5) - dsearchn(cmwx_amp(1 : idx)', 0.5));

    end

    % Define time window of analysis
    prune_times = [-500, 2300]; 
    tf_times = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % Get channel of interest idx
        focuschan =  {'Fz', 'F1', 'F2', 'FC1', 'FCz', 'FC2', 'FFC1h', 'FFC2h'};

        % Get index of channel
        channel_idx = [];
        channels = {'Fz', 'F1', 'F2', 'FC1', 'FCz', 'FC2', 'FFC1h', 'FFC2h'};
        for ch = 1 : length(channels)
            channel_idx(end + 1) = find(strcmp({EEG.chanlocs.labels}, channels{ch}));
        end

        % Get convolution data
        d = squeeze(mean(double(squeeze(EEG.data(channel_idx, :, :))), 1));

        % Get convolution length
        convlen = size(d, 1) * size(d, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(length(tf_freqs), convlen);
        for f = 1 : length(tf_freqs)
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        powcube = NaN(length(tf_freqs), size(d, 1), size(d, 2));
        phacube = NaN(length(tf_freqs), size(d, 1), size(d, 2));
        tmp = fft(reshape(d, 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(d, 1), size(d, 2));
            powcube(f, :, :) = abs(as) .^ 2;
            phacube(f, :, :) = angle(as);    
        end

        % Cut edges
        powcube = powcube(:, dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);
        phacube = phacube(:, dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);

        % Apply single trial baseline
        zcube = zeros(size(powcube));
        blidx = dsearchn(tf_times', [-200, 0]');
        for t = 1 : size(powcube, 3)
            d_trial = squeeze(powcube(:, :, t)); % Get trial tfmat
            blvals = squeeze(mean(d_trial(:, blidx(1) : blidx(2)), 2)); % Get baseline
            blstd = std(d_trial(:, blidx(1) : blidx(2)), 0, 2);
            d_trial = bsxfun(@minus, d_trial, blvals);
            zcube(:, :, t) = bsxfun(@rdivide, d_trial, blstd);
        end

        % Save results
        tf_struct = struct();
        tf_struct.subject = subject;
        tf_struct.id = id;
        tf_struct.tf_times = tf_times;
        tf_struct.tf_freqs = tf_freqs;
        tf_struct.fwhmT = obs_fwhmT;
        tf_struct.fwhmF = obs_fwhmF;
        tf_struct.trialinfo = EEG.trialinfo;
        tf_struct.powcube = powcube;
        tf_struct.phacube = phacube;
        tf_struct.zcube = zcube;
        save([PATH_TFDECOMP, subject, '_tf_decomp'], 'tf_struct');

    end % End subject iteration

end % End part1

% Part 2: ---
if ismember('part2', to_execute)

    % Theta traces matrices
    theta_traces_ersp = [];
    theta_traces_erst = [];
    theta_traces_itpc = [];

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load tf data
        load([PATH_TFDECOMP, subject, '_tf_decomp']);

        % Trialinfo columns:
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
        % 11: correct_response
        % 12: response_side
        % 13: rt
        % 14: accuracy
        % 15: log_response_side
        % 16: log_rt
        % 17: log_accuracy
        % 18: position_color
        % 19: position_tilt
        % 20: position_target
        % 21: position_distractor
        % 22: sequence_position
        % 23: sequence_length

        % Drop incorrect and practice blocks
        to_drop = tf_struct.trialinfo(:, 2) <= 4 | tf_struct.trialinfo(:, 17) ~= 1;
        tf_struct.trialinfo(to_drop, :) = [];
        tf_struct.powcube(:, :, to_drop) = [];
        tf_struct.phacube(:, :, to_drop) = []; 
        tf_struct.zcube(:, :, to_drop) = [];

        % Drop non valid sequence trials
        to_drop = tf_struct.trialinfo(:, 22) == 1;
        tf_struct.trialinfo(to_drop, :) = [];
        tf_struct.powcube(:, :, to_drop) = [];
        tf_struct.phacube(:, :, to_drop) = []; 
        tf_struct.zcube(:, :, to_drop) = [];
        
        % Get condition idx
        idx_std_rep = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 0);
        idx_std_swi = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 1);
        idx_bon_rep = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 0);
        idx_bon_swi = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 1);

        % Get minumum n
        min_n = min([length(idx_std_rep),...
                     length(idx_std_swi),...
                     length(idx_bon_rep),...
                     length(idx_bon_swi)]); 

        % Selcect random samples to balance n
        idx_std_rep = randsample(idx_std_rep, min_n);
        idx_std_swi = randsample(idx_std_swi, min_n);
        idx_bon_rep = randsample(idx_bon_rep, min_n);
        idx_bon_swi = randsample(idx_bon_swi, min_n);

        % Get condition general baseline values
        ersp_bl = [-500, -200];
        idx_all = [idx_std_rep,...
                   idx_std_swi,...
                   idx_bon_rep,...
                   idx_bon_swi]; 
        tmp = squeeze(mean(tf_struct.powcube(:, :, idx_all), 3));
        [~, blidx1] = min(abs(tf_struct.tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_struct.tf_times - ersp_bl(2)));
        blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calc ersp
        ersp_std_rep = 10 * log10(bsxfun(@rdivide, mean(tf_struct.powcube(:, :, idx_std_rep), 3), blvals));
        ersp_std_swi = 10 * log10(bsxfun(@rdivide, mean(tf_struct.powcube(:, :, idx_std_swi), 3), blvals));
        ersp_bon_rep = 10 * log10(bsxfun(@rdivide, mean(tf_struct.powcube(:, :, idx_bon_rep), 3), blvals));
        ersp_bon_swi = 10 * log10(bsxfun(@rdivide, mean(tf_struct.powcube(:, :, idx_bon_swi), 3), blvals));

        % Get single trial baseline ersp
        erst_std_rep = squeeze(mean(tf_struct.zcube(:, :, idx_std_rep), 3));
        erst_std_swi = squeeze(mean(tf_struct.zcube(:, :, idx_std_swi), 3));
        erst_bon_rep = squeeze(mean(tf_struct.zcube(:, :, idx_bon_rep), 3));
        erst_bon_swi = squeeze(mean(tf_struct.zcube(:, :, idx_bon_swi), 3));

        % Calc itpc
        itpc_std_rep = abs(squeeze(mean(exp(1i * tf_struct.phacube(:, :, idx_std_rep)), 3)));
        itpc_std_swi = abs(squeeze(mean(exp(1i * tf_struct.phacube(:, :, idx_std_swi)), 3)));
        itpc_bon_rep = abs(squeeze(mean(exp(1i * tf_struct.phacube(:, :, idx_bon_rep)), 3)));
        itpc_bon_swi = abs(squeeze(mean(exp(1i * tf_struct.phacube(:, :, idx_bon_swi)), 3)));

        % Select frqrange
        idx_freqs = tf_struct.tf_freqs >= 4 & tf_struct.tf_freqs <= 7;

        % Save traces
        theta_traces_ersp(s, 1, 1, :) = mean(ersp_std_rep(idx_freqs, :), 1);
        theta_traces_ersp(s, 1, 2, :) = mean(ersp_std_swi(idx_freqs, :), 1);
        theta_traces_ersp(s, 2, 1, :) = mean(ersp_bon_rep(idx_freqs, :), 1);
        theta_traces_ersp(s, 2, 2, :) = mean(ersp_bon_swi(idx_freqs, :), 1);

        theta_traces_erst(s, 1, 1, :) = mean(erst_std_rep(idx_freqs, :), 1);
        theta_traces_erst(s, 1, 2, :) = mean(erst_std_swi(idx_freqs, :), 1);
        theta_traces_erst(s, 2, 1, :) = mean(erst_bon_rep(idx_freqs, :), 1);
        theta_traces_erst(s, 2, 2, :) = mean(erst_bon_swi(idx_freqs, :), 1);

        theta_traces_itpc(s, 1, 1, :) = mean(itpc_std_rep(idx_freqs, :), 1);
        theta_traces_itpc(s, 1, 2, :) = mean(itpc_std_swi(idx_freqs, :), 1);
        theta_traces_itpc(s, 2, 1, :) = mean(itpc_bon_rep(idx_freqs, :), 1);
        theta_traces_itpc(s, 2, 2, :) = mean(itpc_bon_swi(idx_freqs, :), 1);

    end % End subject loop

    % Average
    theta_traces_ersp_ga = squeeze(mean(theta_traces_ersp, 1));
    theta_traces_erst_ga = squeeze(mean(theta_traces_erst, 1));
    theta_traces_itpc_ga = squeeze(mean(theta_traces_itpc, 1));

    % Plot
    figure

    subplot(1, 2, 1)
    pd = squeeze(theta_traces_ersp_ga(1, 1, :));
    plot(tf_struct.tf_times, pd, '-k', 'LineWidth', 2.5);
    hold on;

    pd = squeeze(theta_traces_ersp_ga(1, 2, :));
    plot(tf_struct.tf_times, pd, ':k', 'LineWidth', 2.5);

    pd = squeeze(theta_traces_ersp_ga(2, 1, :));
    plot(tf_struct.tf_times, pd, '-m', 'LineWidth', 2.5);

    pd = squeeze(theta_traces_ersp_ga(2, 2, :));
    plot(tf_struct.tf_times, pd, ':m', 'LineWidth', 2.5);

    subplot(1, 2, 2)
    pd = squeeze(theta_traces_erst_ga(1, 1, :));
    plot(tf_struct.tf_times, pd, '-k', 'LineWidth', 2.5);
    hold on;

    pd = squeeze(theta_traces_erst_ga(1, 2, :));
    plot(tf_struct.tf_times, pd, ':k', 'LineWidth', 2.5);

    pd = squeeze(theta_traces_erst_ga(2, 1, :));
    plot(tf_struct.tf_times, pd, '-m', 'LineWidth', 2.5);

    pd = squeeze(theta_traces_erst_ga(2, 2, :));
    plot(tf_struct.tf_times, pd, ':m', 'LineWidth', 2.5);

    legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi'});
    grid on;

    % Parameterize time win 1
    time_win = [200, 400];
    [~, idx1] = min(abs(tf_struct.tf_times - time_win(1)));
    [~, idx2] = min(abs(tf_struct.tf_times - time_win(2)));
    params = [mean(squeeze(theta_traces_ersp(:, 1, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 1, 2, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 2, idx1 : idx2)), 2)];

    % Perform rmANOVA
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', params(:, 1), params(:, 2), params(:, 3), params(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    res1 = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    res1

    % Parameterize time win 2
    time_win = [1000, 1200];
    [~, idx1] = min(abs(tf_struct.tf_times - time_win(1)));
    [~, idx2] = min(abs(tf_struct.tf_times - time_win(2)));
    params = [mean(squeeze(theta_traces_ersp(:, 1, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 1, 2, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 2, idx1 : idx2)), 2)];

    % Perform rmANOVA
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', params(:, 1), params(:, 2), params(:, 3), params(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    res2 = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    res2


end % End part2