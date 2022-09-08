
% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.0/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_TF_DATA     = '/mnt/data_dump/bocotilt/4_ersp/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip-master/';
PATH_OUT         = '/mnt/data_dump/bocotilt/4_ersp/out/';
PATH_SELF_REPORT = '/mnt/data_dump/bocotilt/0_logfiles/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% SWITCH: Switch parts of script on/off
to_execute = {'part4'};

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Init eeglab
    addpath(PATH_EEGLAB);
    eeglab;

    % Load info
    EEG = pop_loadset('filename', [subject_list{1} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Set complex Morlet wavelet parameters
    n_frq = 20;
    frqrange = [2, 20];
    tfres_range = [600, 300];

    % Set wavelet time
    wtime = -2 : 1 / EEG.srate : 2;

    % Determine fft frqs
    hz = linspace(0, EEG.srate, length(wtime));

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = logspace(log10(frqrange(1)), log10(frqrange(2)), n_frq);
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
    prune_times = [-500, 2000]; 
    tf_times = EEG.times(dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)));

    % Result struct
    chanlocs = EEG.chanlocs;
    ersp_std_rep = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_std_swi = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_bon_rep = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));
    ersp_bon_swi = single(zeros(length(subject_list), EEG.nbchan, length(tf_freqs), length(tf_times)));

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

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

        % Exclude trials
        to_keep = EEG.trialinfo(:, 2) > 4 &...
                  EEG.trialinfo(:, 23) > 1;

        eeg_data = eeg_data(:, :, to_keep);
        EEG.trialinfo = EEG.trialinfo(to_keep, :);
        EEG.trials = sum(to_keep);

        % Get condition idx
        idx_std_rep = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 0;
        idx_std_swi = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 1;
        idx_bon_rep = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 0;
        idx_bon_swi = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 1;

        % Determine minimal number of trials
        min_n = min([sum(idx_std_rep), sum(idx_std_swi), sum(idx_bon_rep), sum(idx_bon_swi)]);
        
        % Draw balanced samples
        idx_std_rep_balanced = randsample(idx_std_rep, min_n);
        idx_std_swi_balanced = randsample(idx_std_swi, min_n);
        idx_bon_rep_balanced = randsample(idx_bon_rep, min_n);
        idx_bon_swi_balanced = randsample(idx_bon_swi, min_n);

        % Loop channels
        parfor ch = 1 : EEG.nbchan

            % Init tf matrices
            powcube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);

            % Talk
            fprintf('\ntf-decomposition | subject %i/%i | channel %i/%i\n', s, length(subject_list), ch, EEG.nbchan);

            % Get component signal
            channel_data = squeeze(eeg_data(ch, :, :));

            % convolution length
            convlen = size(channel_data, 1) * size(channel_data, 2) + size(cmw, 2) - 1;

            % cmw to freq domain and scale
            cmwX = zeros(length(tf_freqs), convlen);
            for f = 1 : length(tf_freqs)
                cmwX(f, :) = fft(cmw(f, :), convlen);
                cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
            end

            % Get TF-power
            tmp = fft(reshape(channel_data, 1, []), convlen);
            for f = 1 : length(tf_freqs)
                as = ifft(cmwX(f, :) .* tmp); 
                as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
                as = reshape(as, EEG.pnts, EEG.trials);
                powcube(f, :, :) = abs(as) .^ 2;   
            end
           
            % Cut edges
            powcube = powcube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2000), :);

            % Get condition general baseline values
            ersp_bl = [-500, -200];
            tmp = squeeze(mean(powcube, 3));
            [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
            [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
            blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

            % Calculate ersp
            ersp_std_rep(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_rep), 3)), blvals)));
            ersp_std_swi(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_swi), 3)), blvals)));
            ersp_bon_rep(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_rep), 3)), blvals)));
            ersp_bon_swi(s, ch, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_swi), 3)), blvals)));

        end % end channel loop

    end % end subject loop

    % Save shit
    save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
    save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
    save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
    save([PATH_TF_DATA, 'ersp_std_rep.mat'], 'ersp_std_rep');
    save([PATH_TF_DATA, 'ersp_std_swi.mat'], 'ersp_std_swi');
    save([PATH_TF_DATA, 'ersp_bon_rep.mat'], 'ersp_bon_rep');
    save([PATH_TF_DATA, 'ersp_bon_swi.mat'], 'ersp_bon_swi');

end % End part1

% Part 2: Analysis
if ismember('part2', to_execute)

    % Init ft
    addpath(PATH_FIELDTRIP);
    ft_defaults;

    % Load shit
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);
    load([PATH_TF_DATA, 'ersp_std_rep.mat']);
    load([PATH_TF_DATA, 'ersp_std_swi.mat']);
    load([PATH_TF_DATA, 'ersp_bon_rep.mat']);
    load([PATH_TF_DATA, 'ersp_bon_swi.mat']);

    % Exclude
    to_exclude = {};
    idx_exclude = [];
    for ex = 1 : numel(to_exclude)
        idx_exclude(ex) = find(strcmpi(subject_list, to_exclude{ex}));
    end
    ersp_std_rep(idx_exclude, :, :, :) = [];
    ersp_std_swi(idx_exclude, :, :, :) = [];
    ersp_bon_rep(idx_exclude, :, :, :) = [];
    ersp_bon_swi(idx_exclude, :, :, :) = [];

    % Get dims
    [n_subjects, n_channels, n_freqs, n_times] = size(ersp_std_rep);

    % Build elec struct
    for ch = 1 : n_channels
        elec.label{ch} = chanlocs(ch).labels;
        elec.elecpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
        elec.chanpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
    end

    % Save elec for later analyses
    save([PATH_OUT 'elec.mat'], 'elec');

    % Prepare layout
    cfg      = [];
    cfg.elec = elec;
    cfg.rotate = 90;
    layout = ft_prepare_layout(cfg);

    % Re-organize data
    for s = 1 : n_subjects

        ersp_std.powspctrm = double((squeeze(ersp_std_rep(s, :, :, :)) + squeeze(ersp_std_swi(s, :, :, :))) / 2);
        ersp_std.dimord    = 'chan_freq_time';
        ersp_std.label     = elec.label;
        ersp_std.freq      = tf_freqs;
        ersp_std.time      = tf_times;

        ersp_bon.powspctrm = double((squeeze(ersp_bon_rep(s, :, :, :)) + squeeze(ersp_bon_swi(s, :, :, :))) / 2);
        ersp_bon.dimord    = 'chan_freq_time';
        ersp_bon.label     = elec.label;
        ersp_bon.freq      = tf_freqs;
        ersp_bon.time      = tf_times;

        ersp_rep.powspctrm = double((squeeze(ersp_std_rep(s, :, :, :)) + squeeze(ersp_bon_rep(s, :, :, :))) / 2);
        ersp_rep.dimord    = 'chan_freq_time';
        ersp_rep.label     = elec.label;
        ersp_rep.freq      = tf_freqs;
        ersp_rep.time      = tf_times;

        ersp_swi.powspctrm = double((squeeze(ersp_std_swi(s, :, :, :)) + squeeze(ersp_bon_swi(s, :, :, :))) / 2);
        ersp_swi.dimord    = 'chan_freq_time';
        ersp_swi.label     = elec.label;
        ersp_swi.freq      = tf_freqs;
        ersp_swi.time      = tf_times;

        ersp_diff_std.powspctrm = double(squeeze(ersp_std_rep(s, :, :, :)) - squeeze(ersp_std_swi(s, :, :, :)));
        ersp_diff_std.dimord    = 'chan_freq_time';
        ersp_diff_std.label     = elec.label;
        ersp_diff_std.freq      = tf_freqs;
        ersp_diff_std.time      = tf_times;

        ersp_diff_bon.powspctrm = double(squeeze(ersp_bon_rep(s, :, :, :)) - squeeze(ersp_bon_swi(s, :, :, :)));
        ersp_diff_bon.dimord    = 'chan_freq_time';
        ersp_diff_bon.label     = elec.label;
        ersp_diff_bon.freq      = tf_freqs;
        ersp_diff_bon.time      = tf_times;

        d_ersp_std{s} = ersp_std;
        d_ersp_bon{s} = ersp_bon;
        d_ersp_rep{s} = ersp_rep;
        d_ersp_swi{s} = ersp_swi;
        d_ersp_diff_std{s} = ersp_diff_std;
        d_ersp_diff_bon{s} = ersp_diff_bon;

    end

    % Calculate grand averages
    cfg = [];
    cfg.keepindividual = 'yes';
    GA_std = ft_freqgrandaverage(cfg, d_ersp_std{1, :});
    GA_bon = ft_freqgrandaverage(cfg, d_ersp_bon{1, :});
    GA_rep = ft_freqgrandaverage(cfg, d_ersp_rep{1, :});
    GA_swi = ft_freqgrandaverage(cfg, d_ersp_swi{1, :});
    GA_diff_std = ft_freqgrandaverage(cfg, d_ersp_diff_std{1, :});
    GA_diff_bon = ft_freqgrandaverage(cfg, d_ersp_diff_bon{1, :});


    % Define neighbours
    cfg                 = [];
    cfg.layout          = layout;
    cfg.feedback        = 'no';
    cfg.method          = 'triangulation'; 
    cfg.neighbours      = ft_prepare_neighbours(cfg, GA_std);
    neighbours          = cfg.neighbours;

    % Save neighbors for later analyses
    save([PATH_OUT 'neighbours.mat'], 'neighbours');

    % Testparams
    testalpha   = 0.025;
    voxelalpha  = 0.01;
    nperm       = 1000;

    % Set config. Same for all tests
    cfg = [];
    cfg.tail             = 0;
    cfg.statistic        = 'depsamplesT';
    cfg.alpha            = testalpha;
    cfg.neighbours       = neighbours;
    cfg.minnbchan        = 2;
    cfg.method           = 'montecarlo';
    cfg.correctm         = 'cluster';
    cfg.clustertail      = 0;
    cfg.clusteralpha     = voxelalpha;
    cfg.clusterstatistic = 'maxsum';
    cfg.numrandomization = nperm;
    cfg.computecritval   = 'yes'; 
    cfg.ivar             = 1;
    cfg.uvar             = 2;
    cfg.design           = [ones(1, n_subjects), 2 * ones(1, n_subjects); 1 : n_subjects, 1 : n_subjects];

    % The tests
    [stat_bonus] = ft_freqstatistics(cfg, GA_std, GA_bon);
    [stat_switch] = ft_freqstatistics(cfg, GA_rep, GA_swi);
    [stat_interaction] = ft_freqstatistics(cfg, GA_diff_std, GA_diff_bon);

    % Calculate and save effect sizes
    adjpetasq_bonus = [];
    adjpetasq_switch = [];
    adjpetasq_interaction = [];
    for ch = 1 : n_channels
        petasq = (squeeze(stat_bonus.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_bonus.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_bonus(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_switch.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_switch.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_switch(ch, :, :) = adj_petasq;

        petasq = (squeeze(stat_interaction.stat(ch, :, :)) .^ 2) ./ ((squeeze(stat_interaction.stat(ch, :, :)) .^ 2) + (n_subjects - 1));
        adj_petasq = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
        adjpetasq_interaction(ch, :, :) = adj_petasq;
    end

    % Save cluster struct
    save([PATH_OUT 'adjpetasq_bonus.mat'], 'adjpetasq_bonus');
    save([PATH_OUT 'adjpetasq_switch.mat'], 'adjpetasq_switch');
    save([PATH_OUT 'adjpetasq_interaction.mat'], 'adjpetasq_interaction');

    % Identify significant clusters
    clust_thresh = 0.025;
    clusts = struct();
    cnt = 0;
    stat_names = {'stat_bonus', 'stat_switch', 'stat_interaction'};
    for s = 1 : numel(stat_names)
        stat = eval(stat_names{s});
        if ~isempty(stat.negclusters)
            neg_idx = find([stat.negclusters(1, :).prob] < clust_thresh);
            for c = 1 : numel(neg_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = -1;
                clusts(cnt).prob = stat.negclusters(1, neg_idx(c)).prob;
                clusts(cnt).idx = stat.negclusterslabelmat == neg_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat * -1;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2,3])));
            end
        end
        if ~isempty(stat.posclusters)
            pos_idx = find([stat.posclusters(1, :).prob] < clust_thresh);
            for c = 1 : numel(pos_idx)
                cnt = cnt + 1;
                clusts(cnt).testlabel = stat_names{s};
                clusts(cnt).clustnum = cnt;
                clusts(cnt).time = stat.time;
                clusts(cnt).freq = stat.freq;
                clusts(cnt).polarity = 1;
                clusts(cnt).prob = stat.posclusters(1, pos_idx(c)).prob;
                clusts(cnt).idx = stat.posclusterslabelmat == pos_idx(c);
                clusts(cnt).stats = clusts(cnt).idx .* stat.stat;
                clusts(cnt).chans_sig = find(logical(mean(clusts(cnt).idx, [2, 3])));
            end
        end
    end

    % Save cluster struct
    save([PATH_OUT 'significant_clusters.mat'], 'clusts');

    % Plot identified cluster
    clinecol = 'k';
    cmap = 'jet';
    for cnt = 1 : numel(clusts)

        figure('Visible', 'off'); clf;

        subplot(2, 2, 1)
        pd = squeeze(sum(clusts(cnt).stats, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-max(abs(pd(:))), max(abs(pd(:)))], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
        colorbar;
        title(['sum t across chans, plrt: ' num2str(clusts(cnt).polarity)], 'FontSize', 10)

        subplot(2, 2, 2)
        pd = squeeze(mean(clusts(cnt).idx, 1));
        contourf(clusts(cnt).time, clusts(cnt).freq, pd, 40, 'linecolor','none')
        hold on
        contour(clusts(cnt).time, clusts(cnt).freq, logical(squeeze(mean(clusts(cnt).idx, 1))), 1, 'linecolor', clinecol, 'LineWidth', 2)
        colormap(cmap)
        set(gca, 'xlim', [clusts(cnt).time(1), clusts(cnt).time(end)], 'clim', [-1, 1], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
        colorbar;
        title(['proportion chans significant'], 'FontSize', 10)

        subplot(2, 2, 3)
        pd = squeeze(sum(clusts(cnt).stats, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-max(abs(pd(:))), max(abs(pd(:)))])
        colorbar;
        title(['sum t per electrode'], 'FontSize', 10)

        subplot(2, 2, 4)
        pd = squeeze(mean(clusts(cnt).idx, [2, 3]));
        topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'on');
        colormap(cmap)
        set(gca, 'clim', [-1, 1])
        colorbar;
        title(['proportion tf-points significant'], 'FontSize', 10)

        saveas(gcf, [PATH_OUT 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 
    end

end % End part 2


% Part 3: Some plotting
if ismember('part3', to_execute)

    % Init eeglab (for topoplot)
    addpath(PATH_EEGLAB);
    eeglab;

    % Load all the things
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);
    load([PATH_TF_DATA, 'ersp_std_rep.mat']);
    load([PATH_TF_DATA, 'ersp_std_swi.mat']);
    load([PATH_TF_DATA, 'ersp_bon_rep.mat']);
    load([PATH_TF_DATA, 'ersp_bon_swi.mat']);
    load([PATH_OUT, 'significant_clusters.mat']);
    load([PATH_OUT 'adjpetasq_bonus.mat']);
    load([PATH_OUT 'adjpetasq_switch.mat']);
    load([PATH_OUT 'adjpetasq_interaction.mat']);

    % =============== Main effect bonus ========================================================
    ersp_std = squeeze(mean(double((ersp_std_rep + ersp_std_swi) / 2), [1, 2]));
    ersp_bon = squeeze(mean(double((ersp_bon_rep + ersp_bon_swi) / 2), [1, 2]));
    apes_bon = squeeze(mean(adjpetasq_bonus, 1));
    outline_bon = logical(squeeze(mean(clusts(1).idx, 1)) + squeeze(mean(clusts(2).idx, 1)));

    writematrix(ersp_std, [PATH_OUT, 'ersp_std.csv']);
    writematrix(ersp_bon, [PATH_OUT, 'ersp_bon.csv']);
    writematrix(apes_bon, [PATH_OUT, 'apes_bon.csv']);
    writematrix(outline_bon, [PATH_OUT, 'outline_bon.csv']);

    % Plot cluster 1 topo
    idx_time = logical(squeeze(mean(clusts(1).idx, [1, 2])));
    idx_freq = logical(squeeze(mean(clusts(1).idx, [1, 3])));
    idx_chan = logical(squeeze(mean(clusts(1).idx, [2, 3])));
    pd = squeeze(mean(adjpetasq_bonus(:, idx_freq, idx_time), [2, 3]));
    figure('Visible', 'off'); clf;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {find(idx_chan), '.', 'k'} );
    colormap('jet')
    set(gca, 'clim', [-0.3, 0.3])
    saveas(gcf, [PATH_OUT, 'topo_cluster_1.png']);

    % Plot cluster 2 topo
    idx_time = logical(squeeze(mean(clusts(2).idx, [1, 2])));
    idx_freq = logical(squeeze(mean(clusts(2).idx, [1, 3])));
    idx_chan = logical(squeeze(mean(clusts(2).idx, [2, 3])));
    pd = squeeze(mean(adjpetasq_bonus(:, idx_freq, idx_time), [2, 3]));
    figure('Visible', 'off'); clf;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {find(idx_chan), '.', 'k'} );
    colormap('jet')
    set(gca, 'clim', [-0.3, 0.3])
    saveas(gcf, [PATH_OUT, 'topo_cluster_2.png']);

    % =============== Main effect task-switch ========================================================
    ersp_rep = squeeze(mean(double((ersp_std_rep + ersp_bon_rep) / 2), [1, 2]));
    ersp_swi = squeeze(mean(double((ersp_std_swi + ersp_bon_swi) / 2), [1, 2]));
    apes_swi = squeeze(mean(adjpetasq_switch, 1));
    outline_swi = logical(squeeze(mean(clusts(3).idx, 1)) + squeeze(mean(clusts(4).idx, 1)));

    writematrix(ersp_rep, [PATH_OUT, 'ersp_rep.csv']);
    writematrix(ersp_swi, [PATH_OUT, 'ersp_swi.csv']);
    writematrix(apes_swi, [PATH_OUT, 'apes_swi.csv']);
    writematrix(outline_swi, [PATH_OUT, 'outline_swi.csv']);

    % Plot cluster 3 topo
    idx_time = logical(squeeze(mean(clusts(3).idx, [1, 2])));
    idx_freq = logical(squeeze(mean(clusts(3).idx, [1, 3])));
    idx_chan = logical(squeeze(mean(clusts(3).idx, [2, 3])));
    pd = squeeze(mean(adjpetasq_switch(:, idx_freq, idx_time), [2, 3]));
    figure('Visible', 'off'); clf;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {find(idx_chan), '.', 'k'} );
    colormap('jet')
    set(gca, 'clim', [-0.3, 0.3])
    saveas(gcf, [PATH_OUT, 'topo_cluster_3.png']);

    % Plot cluster 4 topo
    idx_time = logical(squeeze(mean(clusts(4).idx, [1, 2])));
    idx_freq = logical(squeeze(mean(clusts(4).idx, [1, 3])));
    idx_chan = logical(squeeze(mean(clusts(4).idx, [2, 3])));
    pd = squeeze(mean(adjpetasq_switch(:, idx_freq, idx_time), [2, 3]));
    figure('Visible', 'off'); clf;
    topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {find(idx_chan), '.', 'k'} );
    colormap('jet')
    set(gca, 'clim', [-0.3, 0.3])
    saveas(gcf, [PATH_OUT, 'topo_cluster_4.png']);

    % =============== Interaction ========================================================
    ersp_diff_std = squeeze(mean(double(ersp_std_rep), [1, 2])) - squeeze(mean(double(ersp_std_swi), [1, 2]));
    ersp_diff_bon = squeeze(mean(double(ersp_bon_rep), [1, 2])) - squeeze(mean(double(ersp_bon_swi), [1, 2]));
    apes_interaction = squeeze(mean(adjpetasq_interaction, 1));

    writematrix(ersp_diff_std, [PATH_OUT, 'ersp_diff_std.csv']);
    writematrix(ersp_diff_bon, [PATH_OUT, 'ersp_diff_bon.csv']);
    writematrix(apes_interaction, [PATH_OUT, 'apes_interaction.csv']);

    % =============== Line plots at FCz and POz ========================================================

    % Some indices...
    idx_frontal = [17, 33, 34, 65, 66, 69, 70, 127];
    idx_posterior = [19, 37, 38, 63, 71, 72, 99, 100];
    idx_delta = tf_freqs < 4;
    idx_theta = tf_freqs >= 4 & tf_freqs <= 8;
    idx_alpha = tf_freqs >= 8 & tf_freqs <= 12;
    idx_beta = tf_freqs >= 14;

    % Save time
    writematrix(tf_times, [PATH_OUT, 'tf_times.csv']);

    % Get lineplots
    fcz_delta = zeros(length(tf_times), 4);
    fcz_delta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_frontal, idx_delta, :), [1, 2, 3]));
    fcz_delta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_frontal, idx_delta, :), [1, 2, 3]));
    fcz_delta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_frontal, idx_delta, :), [1, 2, 3]));
    fcz_delta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_frontal, idx_delta, :), [1, 2, 3]));
    writematrix(fcz_delta, [PATH_OUT, 'lineplots_fcz_delta.csv']);

    fcz_theta = zeros(length(tf_times), 4);
    fcz_theta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    fcz_theta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    fcz_theta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    fcz_theta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_frontal, idx_theta, :), [1, 2, 3]));
    writematrix(fcz_theta, [PATH_OUT, 'lineplots_fcz_theta.csv']);

    fcz_alpha = zeros(length(tf_times), 4);
    fcz_alpha(:, 1) = squeeze(mean(ersp_std_rep(:, idx_frontal, idx_alpha, :), [1, 2, 3]));
    fcz_alpha(:, 2) = squeeze(mean(ersp_std_swi(:, idx_frontal, idx_alpha, :), [1, 2, 3]));
    fcz_alpha(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_frontal, idx_alpha, :), [1, 2, 3]));
    fcz_alpha(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_frontal, idx_alpha, :), [1, 2, 3]));
    writematrix(fcz_alpha, [PATH_OUT, 'lineplots_fcz_alpha.csv']);

    fcz_beta = zeros(length(tf_times), 4);
    fcz_beta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_frontal, idx_beta, :), [1, 2, 3]));
    fcz_beta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_frontal, idx_beta, :), [1, 2, 3]));
    fcz_beta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_frontal, idx_beta, :), [1, 2, 3]));
    fcz_beta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_frontal, idx_beta, :), [1, 2, 3]));
    writematrix(fcz_beta, [PATH_OUT, 'lineplots_fcz_beta.csv']);

    poz_delta = zeros(length(tf_times), 4);
    poz_delta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_posterior, idx_delta, :), [1, 2, 3]));
    poz_delta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_posterior, idx_delta, :), [1, 2, 3]));
    poz_delta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_posterior, idx_delta, :), [1, 2, 3]));
    poz_delta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_posterior, idx_delta, :), [1, 2, 3]));
    writematrix(poz_delta, [PATH_OUT, 'lineplots_poz_delta.csv']);

    poz_theta = zeros(length(tf_times), 4);
    poz_theta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_posterior, idx_theta, :), [1, 2, 3]));
    poz_theta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_posterior, idx_theta, :), [1, 2, 3]));
    poz_theta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_posterior, idx_theta, :), [1, 2, 3]));
    poz_theta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_posterior, idx_theta, :), [1, 2, 3]));
    writematrix(poz_theta, [PATH_OUT, 'lineplots_poz_theta.csv']);

    poz_alpha = zeros(length(tf_times), 4);
    poz_alpha(:, 1) = squeeze(mean(ersp_std_rep(:, idx_posterior, idx_alpha, :), [1, 2, 3]));
    poz_alpha(:, 2) = squeeze(mean(ersp_std_swi(:, idx_posterior, idx_alpha, :), [1, 2, 3]));
    poz_alpha(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_posterior, idx_alpha, :), [1, 2, 3]));
    poz_alpha(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_posterior, idx_alpha, :), [1, 2, 3]));
    writematrix(poz_alpha, [PATH_OUT, 'lineplots_poz_alpha.csv']);

    poz_beta = zeros(length(tf_times), 4);
    poz_beta(:, 1) = squeeze(mean(ersp_std_rep(:, idx_posterior, idx_beta, :), [1, 2, 3]));
    poz_beta(:, 2) = squeeze(mean(ersp_std_swi(:, idx_posterior, idx_beta, :), [1, 2, 3]));
    poz_beta(:, 3) = squeeze(mean(ersp_bon_rep(:, idx_posterior, idx_beta, :), [1, 2, 3]));
    poz_beta(:, 4) = squeeze(mean(ersp_bon_swi(:, idx_posterior, idx_beta, :), [1, 2, 3]));
    writematrix(poz_beta, [PATH_OUT, 'lineplots_poz_beta.csv']);

end % End part 3

% Part 4: Behavioral analysis
if ismember('part4', to_execute)

    % Init eeglab
    addpath(PATH_EEGLAB);
    eeglab;

    % Loop subjects
    behavior_rt = [];
    behavior_ac = [];
    behavior_pp = [];
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

                counter = counter + 1;

                % Get condition idx
                idx_condition = EEG.trialinfo(:, 4) == bon - 1 & EEG.trialinfo(:, 10) == swi - 1;

                % Get correct_idx for condition
                idx_correct = EEG.trialinfo(:, 18) == 1 & idx_condition;

                % Get correct_idx for condition
                idx_points_earned = EEG.trialinfo(:, 24) == 1 & idx_condition;

                % Get accuracy
                ac = sum(idx_correct) / sum(idx_condition);

                % Get rt
                rt = mean(EEG.trialinfo(idx_correct, 15));

                % % of points earned
                pp = sum(idx_points_earned) / sum(idx_condition);


                behavior_rt(s, counter) = rt;
                behavior_ac(s, counter) = ac; 
                behavior_pp(s, counter) = pp; 
 
            end
        end
    end

    % Perform rmANOVA for rt
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', behavior_rt(:, 1), behavior_rt(:, 2), behavior_rt(:, 3), behavior_rt(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    anova_rt = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    anova_rt

    % Perform rmANOVA for accuracy
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', behavior_ac(:, 1), behavior_ac(:, 2), behavior_ac(:, 3), behavior_ac(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    anova_ac = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    anova_ac

    % Perform rmANOVA for points earned
    varnames = {'id', 'b1', 'b2', 'b3', 'b4'};
    t = table([1 : numel(subject_list)]', behavior_pp(:, 1), behavior_pp(:, 2), behavior_pp(:, 3), behavior_pp(:, 4), 'VariableNames', varnames);
    within = table({'std'; 'std'; 'bon'; 'bon'}, {'rep'; 'swi'; 'rep'; 'swi'}, 'VariableNames', {'bonus', 'switch'});
    rm = fitrm(t, 'b1-b4~1', 'WithinDesign', within);
    anova_pp = ranova(rm, 'WithinModel', 'bonus + switch + bonus*switch');
    anova_pp

    % Save behavioral data for veusz
    rt_mean = mean(behavior_rt, 1);
    rt_sd = std(behavior_rt, [], 1) / sqrt(length(subject_list));
    rt_out = [rt_mean(1), rt_sd(1), rt_mean(3), rt_sd(3); rt_mean(2), rt_sd(2), rt_mean(4), rt_sd(4)];
    dlmwrite([PATH_OUT, 'rt_veusz.csv'], rt_out, 'delimiter', '\t');

    ac_mean = mean(behavior_ac, 1);
    ac_sd = std(behavior_ac, [], 1) / sqrt(length(subject_list));
    ac_out = [ac_mean(1), ac_sd(1), ac_mean(3), ac_sd(3); ac_mean(2), ac_sd(2), ac_mean(4), ac_sd(4)];
    dlmwrite([PATH_OUT, 'ac_veusz.csv'], ac_out, 'delimiter', '\t');

    pp_mean = mean(behavior_pp, 1);
    pp_sd = std(behavior_pp, [], 1) / sqrt(length(subject_list));
    pp_out = [pp_mean(1), pp_sd(1), pp_mean(3), pp_sd(3); pp_mean(2), pp_sd(2), pp_mean(4), pp_sd(4)];
    dlmwrite([PATH_OUT, 'pp_veusz.csv'], pp_out, 'delimiter', '\t');

    dlmwrite([PATH_OUT, 'xax.csv'], [1; 2]);


end






















































% Part 99: Correlations
if ismember('part99', to_execute)

    % Collect ids
    id_list = [];
    for s = 1 : length(subject_list)
        id_list(end + 1) = str2num(subject_list{s}(3 : 4));
    end

    % Load all the things
    load([PATH_TF_DATA, 'chanlocs.mat']);
    load([PATH_TF_DATA, 'tf_freqs.mat']);
    load([PATH_TF_DATA, 'tf_times.mat']);
    load([PATH_TF_DATA, 'ersp_std_rep.mat']);
    load([PATH_TF_DATA, 'ersp_std_swi.mat']);
    load([PATH_TF_DATA, 'ersp_bon_rep.mat']);
    load([PATH_TF_DATA, 'ersp_bon_swi.mat']);
    load([PATH_OUT, 'significant_clusters.mat']);
    load([PATH_OUT 'neighbours.mat']);
    load([PATH_OUT 'elec.mat']);

    % Load self report measures
    % 9 columns: 1=id, 2=correct questionnaire version, 3=cond_attended, 4=mot_std, 5=mot_bonus, 6=effort_std, 7=effort_bonus, 8=mw_std, 9=mw_bonus   
    self_reports = readmatrix([PATH_SELF_REPORT, 'self_report_measures.csv']);

    % Get ersp difference as bonus minus standard
    ersp_diff = double((ersp_bon_rep + ersp_bon_swi) / 2) - double((ersp_std_rep + ersp_std_swi) / 2);

    % Get dims
    [n_subjects, n_channels, n_freqs, n_times] = size(ersp_diff);

    % Define frontal and posterior sensor patches
    idx_frontal = [17, 33, 34, 65, 66, 69, 70, 127];
    idx_parietal = [19, 37, 38, 63, 71, 72, 99, 100];

    % Get ERSP for frontal and parietal patches
    ersp_diff_frontal = squeeze(mean(ersp_diff(:, idx_frontal, :, :), 2));
    ersp_diff_parietal = squeeze(mean(ersp_diff(:, idx_parietal, :, :), 2));

    % Build vectors for self report data
    design_condat = [];
    design_motiva = [];
    design_effort = [];
    design_mindwa = [];
    for s = 1 : length(id_list)
        sr_idx = self_reports(:, 1) == id_list(s);
        design_condat(end + 1) = self_reports(sr_idx, 3);                              % Attended to reward condition
        design_motiva(end + 1) = self_reports(sr_idx, 5) - self_reports(sr_idx, 4);    % Difference motivation
        design_effort(end + 1) = self_reports(sr_idx, 7) - self_reports(sr_idx, 6);    % Difference effort
        design_mindwa(end + 1) = self_reports(sr_idx, 9) - self_reports(sr_idx, 8);    % Difference mind wandering
    end

    % Get 2d reüresentation of ERSP data
    ersp_diff_frontal_2d = reshape(ersp_diff_frontal, n_subjects, n_freqs * n_times);
    ersp_diff_parietal_2d = reshape(ersp_diff_parietal, n_subjects, n_freqs * n_times);

    % Group data
    all_ersp_data = {ersp_diff_frontal_2d, ersp_diff_parietal_2d};
    all_self_report_data = {design_condat, design_motiva, design_effort, design_mindwa};

    % Loop!!!
    figure()
    counter = 0;
    for loop1 = 1 : 2
        for loop2 = 1 : 4

            % Select data
            self_report_data = all_self_report_data{loop2}';
            ersp_data = all_ersp_data{loop1};

            % Labels!!!
            ersp_labels = {'frontal', 'parietal'};
            self_report_labels = {'condat', 'motiva', 'effort', 'mindwa'};

            % Get permutation distribution
            n_perms = 1000;
            minmax = zeros(n_perms, 2);
            for np = 1 : n_perms

                % Correlate
                rho_permuted = reshape(corr(self_report_data(randperm(length(self_report_data))), ersp_data), n_freqs, n_times);

                % Get p values
                t_value = (rho_permuted .* sqrt(length(self_report_data) - 2)) ./ sqrt(ones(size(rho_permuted)) - rho_permuted.^2);
                p_values = 1 - tcdf(abs(t_value), length(self_report_data) - 2);

                % Set pixel threshold
                outline = p_values < 0.05;

                % Get cluster test statistics
                clusts = bwconncomp(outline); 
                sum_rho = [];
                cluster_idx = {};
                for clu = 1 : numel(clusts.PixelIdxList)
                    cidx = clusts.PixelIdxList{clu};
                    cluster_idx{clu} = cidx;
                    sum_rho(end + 1) = sum(rho_permuted(cidx));
                end

                % Save min/max
                minmax(np, :) = [min(sum_rho), max(sum_rho)];

            end

            % Get correlation for observed data
            rho_to_test = reshape(corr(self_report_data, ersp_data), n_freqs, n_times);
            
            % Get p-values
            t_value = (rho_to_test .* sqrt(length(self_report_data) - 2)) ./ sqrt(ones(size(rho_to_test)) - rho_to_test.^2);
            p_values = 1 - tcdf(abs(t_value), length(self_report_data) - 2);

            % Set pixel threshold
            outline = p_values < 0.05;
            outline_raw = outline;

            % Get cluster test statistic
            clusts = bwconncomp(outline); 
            sum_rho = [];
            cluster_idx = {};
            for clu = 1 : numel(clusts.PixelIdxList)
                cidx = clusts.PixelIdxList{clu};
                cluster_idx{clu} = cidx;
                sum_rho(end + 1) = sum(rho_to_test(cidx));
            end

            % Remove non-significant clusters
            clust_thresh_lower = prctile(minmax(:, 1), 0.025 * 100);
            clust_thresh_upper = prctile(minmax(:, 2), 100 - 0.025 * 100);
            clust2remove = find(sum_rho > clust_thresh_lower & sum_rho < clust_thresh_upper);
            for clu = 1 : length(clust2remove)
                outline(clusts.PixelIdxList{clust2remove(clu)}) = 0;
            end
            cluster_idx(clust2remove) = [];
            sum_rho(clust2remove) = [];

            % Plot
            counter = counter + 1;
            subplot(2, 4, counter)
            contourf(tf_times, tf_freqs, rho_to_test, 40, 'linecolor','none')
            hold on
            contour(tf_times, tf_freqs, outline, 1, 'linecolor', 'k', 'LineWidth', 2)
            contour(tf_times, tf_freqs, outline_raw, 1, 'linecolor', 'm', 'LineWidth', 2)
            set(gca, 'clim', [-1, 1], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
            colormap('jet')
            colorbar()
            title([ersp_labels{loop1}, ' - ', self_report_labels{loop2}])

        end
    end






end % End part 99