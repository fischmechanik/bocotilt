% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_AUTOCLEANED = '/mnt/data2/bocotilt/2_autocleaned/';
PATH_TF_DATA     = '/mnt/data2/bocotilt/4_ersp/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31'};


%subject_list = {'VP09', 'VP17', 'VP25', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27'};

% SWITCH: Switch parts of script on/off
to_execute = {'part2'};

% Part 1: Calculate ersp
if ismember('part1', to_execute)

    % Init eeglab
    addpath(PATH_EEGLAB);
    eeglab;

    % Load info
    EEG = pop_loadset('filename', [subject_list{1} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

    % Set complex Morlet wavelet parameters
    n_frq = 50;
    frqrange = [2, 25];
    tfres_range = [400, 100];

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = linspace(frqrange(1), frqrange(2), n_frq);
    fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

    % Create wavelets
    wtime = -2 : 1 / EEG.srate : 2;
    for frq = 1 : length(tf_freqs)
        cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);
        cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));
    end

    % Get tf times
    tf_times = EEG.times(dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2000));

    % Result struct
    chanlocs = EEG.chanlocs;
    ersp = zeros(length(subject_list), 2, 2, EEG.nbchan, length(tf_freqs), length(tf_times));
    itpc = zeros(length(subject_list), 2, 2, EEG.nbchan, length(tf_freqs), length(tf_times));

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

        % Exclude trials
        to_keep = EEG.trialinfo(:, 2) > 4 & EEG.trialinfo(:, 17) == 1 & EEG.trialinfo(:, 22) > 1;
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
        
        % Draw samples
        idx_std_rep = randsample(idx_std_rep, min_n);
        idx_std_swi = randsample(idx_std_swi, min_n);
        idx_bon_rep = randsample(idx_bon_rep, min_n);
        idx_bon_swi = randsample(idx_bon_swi, min_n);

        % Loop channels
        for ch = 1 : EEG.nbchan

            % Talk
            fprintf('\ntf-decomposition | subject %i/%i | channel %i/%i\n', s, length(subject_list), ch, EEG.nbchan);

            % Init tf matrices
            powcube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);
            phacube = NaN(length(tf_freqs), EEG.pnts, EEG.trials);

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
                phacube(f, :, :) = exp(1i * angle(as));     
            end

            % Cut edges
            powcube = powcube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2000), :);
            phacube = phacube(:, dsearchn(EEG.times', -500) : dsearchn(EEG.times', 2000), :);

            % Get condition general baseline values
            bl_idx = tf_times >= -500 & tf_times <= -200;
            tmp = squeeze(mean(powcube, 3));
            blvals = squeeze(mean(tmp(:, bl_idx), 2));

            % Calculate ersp
            ersp(s, 1, 1, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_rep), 3)), blvals));
            ersp(s, 1, 2, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_swi), 3)), blvals));
            ersp(s, 2, 1, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_rep), 3)), blvals));
            ersp(s, 2, 2, ch, :, :) = 10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_swi), 3)), blvals));

            % Calculate itpc
            itpc(s, 1, 1, ch, :, :)  = abs(squeeze(mean(phacube(:, :, idx_std_rep), 3)));
            itpc(s, 1, 2, ch, :, :)  = abs(squeeze(mean(phacube(:, :, idx_std_rep), 3)));
            itpc(s, 2, 1, ch, :, :)  = abs(squeeze(mean(phacube(:, :, idx_std_rep), 3)));
            itpc(s, 2, 2, ch, :, :)  = abs(squeeze(mean(phacube(:, :, idx_std_rep), 3)));

        end % end channel loop
    end % end subject loop

    % Convert to single
    ersp = single(ersp);
    itpc = single(itpc);

    % Save shit
    save([PATH_TF_DATA, 'chanlocs.mat'], 'chanlocs');
    save([PATH_TF_DATA, 'tf_freqs.mat'], 'tf_freqs');
    save([PATH_TF_DATA, 'tf_times.mat'], 'tf_times');
    save([PATH_TF_DATA, 'ersp.mat'], 'ersp');
    save([PATH_TF_DATA, 'itpc.mat'], 'itpc');

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
    load([PATH_TF_DATA, 'ersp.mat']);

    % To double
    ersp = double(ersp);

    % Get dims
    [n_subjects, n_bonus, n_switch, n_channels, n_freqs, n_times] = size(ersp);

    % Build elec struct
    for ch = 1 : n_channels
        elec.label{ch} = chanlocs(ch).labels;
        elec.elecpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
        elec.chanpos(ch, :) = [chanlocs(ch).X, chanlocs(ch).Y, chanlocs(ch).Z];
    end

    % Prepare layout
    cfg      = [];
    cfg.elec = elec;
    cfg.rotate = 90;
    layout = ft_prepare_layout(cfg);


    % Re-organize data
    for s = 1: n_subjects

        ersp_std.powspctrm = (squeeze(ersp(s, 1, 1, :, :, :)) + squeeze(ersp(s, 1, 2, :, :, :))) / 2;
        ersp_std.dimord    = 'chan_freq_time';
        ersp_std.label     = elec.label;
        ersp_std.freq      = tf_freqs;
        ersp_std.time      = tf_times;

        ersp_bon.powspctrm = (squeeze(ersp(s, 2, 1, :, :, :)) + squeeze(ersp(s, 2, 2, :, :, :))) / 2;
        ersp_bon.dimord    = 'chan_freq_time';
        ersp_bon.label     = elec.label;
        ersp_bon.freq      = tf_freqs;
        ersp_bon.time      = tf_times;

        ersp_rep.powspctrm = (squeeze(ersp(s, 1, 1, :, :, :)) + squeeze(ersp(s, 2, 1, :, :, :))) / 2;
        ersp_rep.dimord    = 'chan_freq_time';
        ersp_rep.label     = elec.label;
        ersp_rep.freq      = tf_freqs;
        ersp_rep.time      = tf_times;

        ersp_swi.powspctrm = (squeeze(ersp(s, 1, 2, :, :, :)) + squeeze(ersp(s, 2, 2, :, :, :))) / 2;
        ersp_swi.dimord    = 'chan_freq_time';
        ersp_swi.label     = elec.label;
        ersp_swi.freq      = tf_freqs;
        ersp_swi.time      = tf_times;

        ersp_diff_std.powspctrm = squeeze(ersp(s, 1, 1, :, :, :)) - squeeze(ersp(s, 1, 2, :, :, :));
        ersp_diff_std.dimord    = 'chan_freq_time';
        ersp_diff_std.label     = elec.label;
        ersp_diff_std.freq      = tf_freqs;
        ersp_diff_std.time      = tf_times;

        ersp_diff_bon.powspctrm = squeeze(ersp(s, 2, 1, :, :, :)) - squeeze(ersp(s, 2, 2, :, :, :));
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

    aa=bb

    % Calculate and save effect sizes
    adjpetasq_bonus = [];
    adjpetasq_switch = [];
    adjpetasq_interaction = [];
    for ch = 1 : EEG.nbchan
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

    % Identify significant clusters
    clusts = struct();
    cnt = 0;
    stat_names = {'stat_bonus', 'stat_switch', 'stat_interaction'};
    for s = 1 : numel(stat_names)
        stat = eval(stat_names{s});
        if ~isempty(stat.negclusters)
            neg_idx = find([stat.negclusters(1, :).prob] < testalpha);
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
            pos_idx = find([stat.posclusters(1, :).prob] < testalpha);
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
    save([PATH_CLUSTSTATS 'significant_clusters.mat'], 'clusts');

    % Plot identified cluster
    clinecol = 'k';
    cmap = 'jet';
    chanlocs = EEG.chanlocs;
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

        saveas(gcf, [PATH_PLOT 'clustnum_' num2str(clusts(cnt).clustnum) '_' clusts(cnt).testlabel '.png']); 
    end



end