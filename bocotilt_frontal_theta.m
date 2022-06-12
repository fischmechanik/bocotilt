% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.0/';
PATH_AUTOCLEANED = '/mnt/data2/bocotilt/2_autocleaned/';
PATH_TFDECOMP    = '/mnt/data2/bocotilt/frontal_theta/';
PATH_LOGFILES    = '/mnt/data2/bocotilt/0_logfiles/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31'};

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

        % Iterate last 100 trials and extract rt thresholds
        rt_threshs = [];
        for l = 1 : 100
            line_values = split(logcell{length(logcell) - l}, ' ');
            rt_threshs(l, 1) = str2num(line_values{5});
            rt_threshs(l, 2) = str2num(line_values{13});
        end
        rt_thresh_color = mean(rt_threshs(rt_threshs(:, 1) == 2, 2));
        rt_thresh_tilt = mean(rt_threshs(rt_threshs(:, 1) == 1, 2));

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
        tf_struct.rt_thresh_color = rt_thresh_color;
        tf_struct.rt_thresh_tilt = rt_thresh_tilt;
        save([PATH_TFDECOMP, subject, '_tf_decomp'], 'tf_struct');

    end % End subject iteration

end % End part1

% Part 2: ---
if ismember('part2', to_execute)

    % Theta traces matrices
    theta_traces_ersp = [];
    theta_traces_erst = [];
    rts = [];
    accuracy = [];
    incorrect = [];
    omissions = [];

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load tf data
        load([PATH_TFDECOMP, subject, '_tf_decomp']);

        % Add to trialinfo
        for t = 1 : size(tf_struct.trialinfo, 1)
            if tf_struct.trialinfo(t, 17) == 1 & tf_struct.trialinfo(t, 5) == 0 & tf_struct.trialinfo(t, 16) <= tf_struct.rt_thresh_color
                tf_struct.trialinfo(t, 24) = 1;
            elseif tf_struct.trialinfo(t, 17) == 1 & tf_struct.trialinfo(t, 5) == 1 & tf_struct.trialinfo(t, 16) <= tf_struct.rt_thresh_tilt
                tf_struct.trialinfo(t, 24) = 1;
            else
                tf_struct.trialinfo(t, 24) = 0;
            end
        end

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
        % 24: Better than thresh

        % Drop practice blocks
        to_drop = tf_struct.trialinfo(:, 2) <= 4;
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

        % Get accuracy measures
        idx_std_rep = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 0);
        idx_std_swi = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 1);
        idx_bon_rep = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 0);
        idx_bon_swi = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 1);

        accuracy(s, 1) =   sum(tf_struct.trialinfo(idx_std_rep, 17) == 1) / size(idx_std_rep, 1);
        incorrect(s, 1) = sum(tf_struct.trialinfo(idx_std_rep, 17) == 0) / size(idx_std_rep, 1);
        omissions(s, 1) =  sum(tf_struct.trialinfo(idx_std_rep, 17) == 2) / size(idx_std_rep, 1);

        accuracy(s, 2) =   sum(tf_struct.trialinfo(idx_std_swi, 17) == 1) / size(idx_std_swi, 1);
        incorrect(s, 2) = sum(tf_struct.trialinfo(idx_std_swi, 17) == 0) / size(idx_std_swi, 1);
        omissions(s, 2) =  sum(tf_struct.trialinfo(idx_std_swi, 17) == 2) / size(idx_std_swi, 1);

        accuracy(s, 3) =   sum(tf_struct.trialinfo(idx_bon_rep, 17) == 1) / size(idx_bon_rep, 1);
        incorrect(s, 3) = sum(tf_struct.trialinfo(idx_bon_rep, 17) == 0) / size(idx_bon_rep, 1);
        omissions(s, 3) =  sum(tf_struct.trialinfo(idx_bon_rep, 17) == 2) / size(idx_bon_rep, 1);

        accuracy(s, 4) =   sum(tf_struct.trialinfo(idx_bon_swi, 17) == 1) / size(idx_bon_swi, 1);
        incorrect(s, 4) = sum(tf_struct.trialinfo(idx_bon_swi, 17) == 0) / size(idx_bon_swi, 1);
        omissions(s, 4) =  sum(tf_struct.trialinfo(idx_bon_swi, 17) == 2) / size(idx_bon_swi, 1);

        % Drop incorrect 
        to_drop = tf_struct.trialinfo(:, 17) ~= 1;
        tf_struct.trialinfo(to_drop, :) = [];
        tf_struct.powcube(:, :, to_drop) = [];
        tf_struct.phacube(:, :, to_drop) = []; 
        tf_struct.zcube(:, :, to_drop) = [];
        
        % Get condition idx again
        idx_std_rep = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 0);
        idx_std_swi = find(tf_struct.trialinfo(:, 4) == 0 & tf_struct.trialinfo(:, 10) == 1);
        idx_bon_rep = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 0);
        idx_bon_swi = find(tf_struct.trialinfo(:, 4) == 1 & tf_struct.trialinfo(:, 10) == 1);

        % Get condition general baseline values
        ersp_bl = [-500, -200];
        tmp = squeeze(mean(tf_struct.powcube, 3));
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

        % Get rt
        rts(s, 1) = mean(tf_struct.trialinfo(idx_std_rep, 16));
        rts(s, 2) = mean(tf_struct.trialinfo(idx_std_swi, 16));
        rts(s, 3) = mean(tf_struct.trialinfo(idx_bon_rep, 16));
        rts(s, 4) = mean(tf_struct.trialinfo(idx_bon_swi, 16));

        % Select frqrange
        idx_freqs = tf_struct.tf_freqs >= 4 & tf_struct.tf_freqs <= 8;

        % Save traces
        theta_traces_ersp(s, 1, 1, :) = mean(ersp_std_rep(idx_freqs, :), 1);
        theta_traces_ersp(s, 1, 2, :) = mean(ersp_std_swi(idx_freqs, :), 1);
        theta_traces_ersp(s, 2, 1, :) = mean(ersp_bon_rep(idx_freqs, :), 1);
        theta_traces_ersp(s, 2, 2, :) = mean(ersp_bon_swi(idx_freqs, :), 1);

        theta_traces_erst(s, 1, 1, :) = mean(erst_std_rep(idx_freqs, :), 1);
        theta_traces_erst(s, 1, 2, :) = mean(erst_std_swi(idx_freqs, :), 1);
        theta_traces_erst(s, 2, 1, :) = mean(erst_bon_rep(idx_freqs, :), 1);
        theta_traces_erst(s, 2, 2, :) = mean(erst_bon_swi(idx_freqs, :), 1);

    end % End subject loop

    % Plot
    figure()
    ylims = [-15, 15];

    subplot(2, 1, 1)
    pd = squeeze(theta_traces_erst(:, 1, 2, :)) - squeeze(theta_traces_erst(:, 1, 1, :));
    plot(tf_struct.tf_times, pd, 'LineWidth', 2.5);
    ylim(ylims)
    title('switch- rep in standard')
    
    subplot(2, 1, 2)
    pd = squeeze(theta_traces_erst(:, 2, 2, :)) - squeeze(theta_traces_erst(:, 2, 1, :));
    plot(tf_struct.tf_times, pd, 'LineWidth', 2.5);
    ylim(ylims)
    title('switch- rep in bonus')


    figure()
    pd = squeeze(mean(squeeze(theta_traces_erst(:, 1, 1, :)), 1));
    plot(tf_struct.tf_times, pd, 'k', 'LineWidth', 2.5);
    hold on
    pd = squeeze(mean(squeeze(theta_traces_erst(:, 1, 2, :)), 1));
    plot(tf_struct.tf_times, pd, ':k', 'LineWidth', 2.5);

    pd = squeeze(mean(squeeze(theta_traces_erst(:, 2, 1, :)), 1));
    plot(tf_struct.tf_times, pd, 'm', 'LineWidth', 2.5);
    pd = squeeze(mean(squeeze(theta_traces_erst(:, 2, 2, :)), 1));
    plot(tf_struct.tf_times, pd, ':m', 'LineWidth', 2.5);
    legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi'})


    % Parameterize time win 1
    time_win = [100, 700];
    [~, idx1] = min(abs(tf_struct.tf_times - time_win(1)));
    [~, idx2] = min(abs(tf_struct.tf_times - time_win(2)));
    params_win1 = [mean(squeeze(theta_traces_erst(:, 1, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_erst(:, 1, 2, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_erst(:, 2, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_erst(:, 2, 2, idx1 : idx2)), 2)];

    % Perform rmANOVA
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', params_win1(:, 1), params_win1(:, 2), params_win1(:, 3), params_win1(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    anova_theta1 = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    anova_theta1

    figure
    subplot(1, 3, 1)
    plot([1,2], params_win1(:, [1,2]), 'LineWidth', 2.5)
    title('std')
    subplot(1, 3, 2)
    plot([1,2], params_win1(:, [3,4]), 'LineWidth', 2.5)
    title('bonus')

    % Estimate strategies based on frontal theta differences between rep and swi
    strats = [];
    for s = 1 : size(params_win1, 1)

        % Both rep more
        if params_win1(s, 1) > params_win1(s, 2) & params_win1(s, 3) > params_win1(s, 4)
            strats(s) = 1;

        % Both switch more
        elseif params_win1(s, 1) < params_win1(s, 2) & params_win1(s, 3) < params_win1(s, 4)
            strats(s) = 2;

        % Only in std more in switch
        elseif params_win1(s, 1) < params_win1(s, 2) & params_win1(s, 3) > params_win1(s, 4)
            strats(s) = 3;

        % Only in bonus more in switch
        elseif params_win1(s, 1) > params_win1(s, 2) & params_win1(s, 3) < params_win1(s, 4)
            strats(s) = 4;
        end

    end

    % Get average behavioral measures for strats
    rt_by_strats = [mean(rts(strats == 1, :), 1);mean(rts(strats == 2, :), 1);mean(rts(strats == 3, :), 1);mean(rts(strats == 4, :), 1)];
    accuracy_by_strats = [mean(accuracy(strats == 1, :), 1);mean(accuracy(strats == 2, :), 1);mean(accuracy(strats == 3, :), 1);mean(accuracy(strats == 4, :), 1)];
    incorrect_by_strats = [mean(incorrect(strats == 1, :), 1);mean(incorrect(strats == 2, :), 1);mean(incorrect(strats == 3, :), 1);mean(incorrect(strats == 4, :), 1)];
    omissions_by_strats = [mean(omissions(strats == 1, :), 1);mean(omissions(strats == 2, :), 1);mean(omissions(strats == 3, :), 1);mean(omissions(strats == 4, :), 1)];

    figure()
    subplot(2, 2, 1)
    scatter(zscore(rts(:, 1)), zscore(accuracy(:, 1)))
    title('std-rep')

    subplot(2, 2, 2)
    scatter(zscore(rts(:, 2)), zscore(accuracy(:, 2)))
    title('std-swi')

    subplot(2, 2, 3)
    scatter(zscore(rts(:, 3)), zscore(accuracy(:, 3)))
    title('bon-rep')

    subplot(2, 2, 4)
    scatter(zscore(rts(:, 4)), zscore(accuracy(:, 4)))
    title('bon-swi')
    aa=bb

    % 
    subj_select = strats == 3 | strats == 4; 
    diffs_rt_std = zscore(rts(subj_select, 2) - rts(subj_select, 1));
    diffs_rt_bon = zscore(rts(subj_select, 4) - rts(subj_select, 3));
    diffs_w1_std = zscore(params_win1(subj_select, 2) - params_win1(subj_select, 1));
    diffs_w1_bon = zscore(params_win1(subj_select, 4) - params_win1(subj_select, 3));

    [r, pval] = corrcoef(diffs_rt_bon, diffs_w1_bon);



    % Parameterize time win 2
    time_win = [1000, 1200];
    [~, idx1] = min(abs(tf_struct.tf_times - time_win(1)));
    [~, idx2] = min(abs(tf_struct.tf_times - time_win(2)));
    params_win2 = [mean(squeeze(theta_traces_ersp(:, 1, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 1, 2, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 1, idx1 : idx2)), 2),...
              mean(squeeze(theta_traces_ersp(:, 2, 2, idx1 : idx2)), 2)];

    % Perform rmANOVA for rt
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', rts(:, 1), rts(:, 2), rts(:, 3), rts(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    anova_rt = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    anova_rt

    % Perform rmANOVA
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', params_win2(:, 1), params_win2(:, 2), params_win2(:, 3), params_win2(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    res2 = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    res2


end % End part2