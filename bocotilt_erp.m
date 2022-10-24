% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_FIELDTRIP   = '/home/plkn/fieldtrip-master/';
PATH_OUTPUT      = '/mnt/data_dump/bocotilt/99_erp_results/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Init ft
addpath(PATH_FIELDTRIP);
ft_defaults;

% This is where we collect the ERPs
erp_matrix = [];

% Loop subjects
for s = 1 : length(subject_list)

    % Load data
    EEG = pop_loadset('filename', [subject_list{s} '_cleaned_erp.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

    % Trial data
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
    
    % The data matrix: channels x times x trials
    eeg_data = double(EEG.data);

    % Prune time
    idx_keep = EEG.times >= -200 & EEG.times <= 1600;
    eeg_times = EEG.times(idx_keep);
    eeg_data = eeg_data(:, idx_keep, :);

    % Exclude trials
    to_keep = EEG.trialinfo(:, 2) > 4 &...
              EEG.trialinfo(:, 23) > 1;
    eeg_data = double(eeg_data(:, :, to_keep));
    EEG.trialinfo = EEG.trialinfo(to_keep, :);

    % Get condition idx
    idx_std_rep = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 0;
    idx_std_swi = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 1;
    idx_bon_rep = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 0;
    idx_bon_swi = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 1;

    % Collect erps
    erp_matrix(s, 1, 1, :, :) = squeeze(mean(eeg_data(:, :, idx_std_rep), 3));
    erp_matrix(s, 1, 2, :, :) = squeeze(mean(eeg_data(:, :, idx_std_swi), 3));
    erp_matrix(s, 2, 1, :, :) = squeeze(mean(eeg_data(:, :, idx_bon_rep), 3));
    erp_matrix(s, 2, 2, :, :) = squeeze(mean(eeg_data(:, :, idx_bon_swi), 3));

end

% % Build elec struct
% for ch = 1 : size(erp_matrix, 4)
%     elec.label{ch} = EEG.chanlocs(ch).labels;
%     elec.elecpos(ch, :) = [EEG.chanlocs(ch).X, EEG.chanlocs(ch).Y, EEG.chanlocs(ch).Z];
%     elec.chanpos(ch, :) = [EEG.chanlocs(ch).X, EEG.chanlocs(ch).Y, EEG.chanlocs(ch).Z];
% end

% % Prepare layout
% cfg      = [];
% cfg.elec = elec;
% cfg.rotate = 90;
% layout = ft_prepare_layout(cfg);

% Build chanlocs
chanlabs = {};
coords = [];
for c = 1 : numel(EEG.chanlocs)
    chanlabs{c} = EEG.chanlocs(c).labels;
    coords(c, :) = [EEG.chanlocs(c).X, EEG.chanlocs(c).Y, EEG.chanlocs(c).Z];
end

 % A sensor struct
 sensors = struct();
 sensors.label = chanlabs;
 sensors.chanpos = coords;
 sensors.elecpos = coords;

 % Prepare neighbor struct
 cfg                 = [];
 cfg.elec            = sensors;
 cfg.feedback        = 'no';
 cfg.method          = 'triangulation'; 
 neighbours          = ft_prepare_neighbours(cfg);

% A template for GA structs
cfg=[];
cfg.keepindividual = 'yes';
d = [];
d.dimord = 'chan_time';
d.label = chanlabs;
d.time = eeg_times;

% Build GA struct standard
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(mean(erp_matrix(s, 1, :, :, :), 3));
    D{s} = d;
end 
GA_std = ft_timelockgrandaverage(cfg, D{1, :});
%GA_std.dimord = 'subj_chan_time';

% Build GA struct bonus
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(mean(erp_matrix(s, 2, :, :, :), 3));
    D{s} = d;
end 
GA_bon = ft_timelockgrandaverage(cfg, D{1, :});
%GA_bon.dimord = 'subj_chan_time';

% Build GA struct repeat
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(mean(erp_matrix(s, :, 1, :, :), 2));
    D{s} = d;
end 
GA_rep = ft_timelockgrandaverage(cfg, D{1, :});

% Build GA struct switch
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(mean(erp_matrix(s, :, 2, :, :), 2));
    D{s} = d;
end 
GA_swi = ft_timelockgrandaverage(cfg, D{1, :});

% Build GA struct switch-repeat in standard
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(erp_matrix(s, 1, 2, :, :)) - squeeze(erp_matrix(s, 1, 1, :, :));
    D{s} = d;
end 
GA_diff_std = ft_timelockgrandaverage(cfg, D{1, :});

% Build GA struct switch-repeat in bonus
D = {};
for s = 1 : size(erp_matrix, 1)
    d.avg = squeeze(erp_matrix(s, 2, 2, :, :)) - squeeze(erp_matrix(s, 2, 1, :, :));
    D{s} = d;
end 
GA_diff_bon = ft_timelockgrandaverage(cfg, D{1, :});

% Cluster permutation tests
a_cluster_test('bonus-vs-standard', GA_bon, GA_std, 'bonus', 'standard', EEG.chanlocs, neighbours, PATH_OUTPUT, PATH_OUTPUT);
a_cluster_test('switch-vs-repeat', GA_swi, GA_rep, 'switch', 'repeat', EEG.chanlocs, neighbours, PATH_OUTPUT, PATH_OUTPUT);
a_cluster_test('interaction', GA_diff_bon, GA_diff_std, 'diffbon', 'diffstd', EEG.chanlocs, neighbours, PATH_OUTPUT, PATH_OUTPUT);

% Plot some midline electrodes
ga_erps = squeeze(mean(erp_matrix, 1));
ga_std_rep = squeeze(ga_erps(1, 1, :, :));
ga_std_swi = squeeze(ga_erps(1, 2, :, :));
ga_bon_rep = squeeze(ga_erps(2, 1, :, :));
ga_bon_swi = squeeze(ga_erps(2, 2, :, :));


figure()

subplot(3, 1, 1)
plot(eeg_times, ga_std_rep(17, :), 'LineWidth', 2, 'Color', [1, 0, 0])
hold on;
plot(eeg_times, ga_std_swi(17, :), 'LineWidth', 2, 'Color', [0, 1, 0])
plot(eeg_times, ga_bon_rep(17, :), 'LineWidth', 2, 'Color', [0, 0, 1])
plot(eeg_times, ga_bon_swi(17, :), 'LineWidth', 2, 'Color', [1, 0, 1])
title('Fz')
legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi'})
xline([0, 800])

subplot(3, 1, 2)
plot(eeg_times, ga_std_rep(18, :), 'LineWidth', 2, 'Color', [1, 0, 0])
hold on;
plot(eeg_times, ga_std_swi(18, :), 'LineWidth', 2, 'Color', [0, 1, 0])
plot(eeg_times, ga_bon_rep(18, :), 'LineWidth', 2, 'Color', [0, 0, 1])
plot(eeg_times, ga_bon_swi(18, :), 'LineWidth', 2, 'Color', [1, 0, 1])
title('Cz')
legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi'})
xline([0, 800])

subplot(3, 1, 3)
plot(eeg_times, ga_std_rep(19, :), 'LineWidth', 2, 'Color', [1, 0, 0])
hold on;
plot(eeg_times, ga_std_swi(19, :), 'LineWidth', 2, 'Color', [0, 1, 0])
plot(eeg_times, ga_bon_rep(19, :), 'LineWidth', 2, 'Color', [0, 0, 1])
plot(eeg_times, ga_bon_swi(19, :), 'LineWidth', 2, 'Color', [1, 0, 1])
title('Pz')
legend({'std-rep', 'std-swi', 'bon-rep', 'bon-swi'})
xline([0, 800])






















% Function that performs test and creates cluster-plots
function[] = a_cluster_test(titlestring, cond1, cond2, cond1string, cond2string, chanlocs, neighbours, PATH_PLOT, PATH_VEUSZ)

    % Create output directory
    mkdir([PATH_PLOT, titlestring])
    PATH_OUTPUT = [PATH_PLOT, titlestring, '/'];

    % Testparams
    testalpha  = 0.025;
    voxelalpha  = 0.01;
    nperm = 1000;

    % Set config
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
    cfg.design           = [ones(1, size(cond1.individual, 1)), 2 * ones(1, size(cond1.individual, 1)); 1 : size(cond1.individual, 1), 1 : size(cond1.individual, 1)];
    
    % The test
    [stat] = ft_timelockstatistics(cfg, cond1, cond2);  

    % Calculate effect sizes
    apes = [];
    n_subjects = size(cond1.individual, 1);
    for ch = 1 : numel(chanlocs)
        petasq = (squeeze(stat.stat(ch, :)) .^ 2) ./ ((squeeze(stat.stat(ch, :)) .^ 2) + (n_subjects - 1));
        apes(ch, :) = petasq - (1 - petasq) .* (1 / (n_subjects - 1));
    end

    % Plot effect sizes
    cmap = 'jet';
    clim = [-0.6, 0.6];
    figure('Visible', 'off'); clf;
    contourf(stat.time, [1 : numel(cond1.label)], apes, 40, 'linecolor','none')
    colormap(cmap)
    set(gca, 'clim', [clim])
    colorbar;
    title(['effect sizes: ', titlestring])
    saveas(gcf, [PATH_OUTPUT, 'effect_sizes_', titlestring, '.png']);

    % Plot conditions
    pd1 = squeeze(mean(cond1.individual, 1));
    pd2 = squeeze(mean(cond2.individual, 1));
    cmap = 'jet';
    lim = max(abs([pd1(:); pd2(:)]));
    clim = [-lim, lim];
    figure('Visible', 'off'); clf;
    subplot(1, 2, 1)
    contourf(stat.time, [1 : numel(cond1.label)], pd1, 40, 'linecolor','none')
    colormap(cmap)
    set(gca, 'clim', [clim])
    title(cond1string)
    subplot(1, 2, 2)
    contourf(stat.time, [1 : numel(cond1.label)], pd2, 40, 'linecolor','none')
    colormap(cmap)
    set(gca, 'clim', [clim])
    title(cond2string)
    saveas(gcf, [PATH_OUTPUT, 'ersp_', titlestring, '.png']);

    % Save effect sizes
    dlmwrite([PATH_VEUSZ, titlestring, '_effect_sizes.csv'], apes);

    % Save averages
    dlmwrite([PATH_VEUSZ, titlestring, '_' cond1string '_average.csv'], squeeze(mean(cond1.individual, 1)));
    dlmwrite([PATH_VEUSZ, titlestring, '_' cond2string '_average.csv'], squeeze(mean(cond2.individual, 1)));
    dlmwrite([PATH_VEUSZ, titlestring, '_difference.csv'], squeeze(mean(cond1.individual, 1)) - squeeze(mean(cond2.individual, 1)));

    % Set threshold to 0.025
    sig_pos = find([stat.posclusters.prob] <= testalpha);
    sig_neg = find([stat.negclusters.prob] <= testalpha);

    % Plot clusters
    if sig_pos
        for cl = 1 : length(sig_pos)

            % Indices of the cluster
            idx = stat.posclusterslabelmat == sig_pos(cl);
            pval = round(stat.posclusters(sig_pos(cl)).prob, 3);

            % Save contour of cluster
            dlmwrite([PATH_VEUSZ, titlestring, '_contour_poscluster_', num2str(sig_pos(cl)), '.csv'], idx);

            % Identify significant channels and time points
            chans_sig = find(sum(idx, 2));
            times_sig = find(sum(idx, 1));

            % Plot a topo of effect sizes
            markercolor = 'k';
            markersize = 10;
            cmap = 'jet';
            clim = [-0.5, 0.5];
            figure('Visible', 'off'); clf;
            pd = mean(apes(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            saveas(gcf, [PATH_OUTPUT, [titlestring, '_effectsize_topo_pos_'], num2str(sig_pos(cl)), '.png']);

            % Plot
            markercolor = 'k';
            markersize = 10;
            cmap = 'jet';
            clim = [-5, 5];
            figure('Visible', 'off'); clf;

            subplot(2, 2, 1)
            pd = squeeze(mean(cond1.individual, 1));
            contourf(stat.time, [1 : numel(cond1.label)], pd, 40, 'linecolor','none')
            hold on
            contour(stat.time, [1 : numel(cond1.label)], idx, 1, 'linecolor', 'k', 'LineWidth', 2)
            colormap(cmap)
            set(gca, 'clim', [clim])
            colorbar;
            title(cond1string)

            subplot(2, 2, 2)
            pd = squeeze(mean(cond2.individual, 1));
            contourf(stat.time, [1 : numel(cond1.label)], pd, 40, 'linecolor','none')
            hold on
            contour(stat.time, [1 : numel(cond1.label)], idx, 1, 'linecolor', 'k', 'LineWidth', 2)
            colormap(cmap)
            set(gca, 'clim', [clim])
            colorbar;
            title(cond2string)
            
            subplot(2, 2, 3)
            pd = squeeze(mean(cond1.individual, 1));
            pd = mean(pd(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            title(['poscluster #' num2str(sig_pos(cl)) ' - p=' num2str(pval)])

            subplot(2, 2, 4)
            pd = squeeze(mean(cond2.individual, 1));
            pd = mean(pd(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            title(['poscluster #' num2str(sig_pos(cl)) ' - p=' num2str(pval)])

            saveas(gcf, [PATH_OUTPUT, [titlestring, '_pos_'], num2str(sig_pos(cl)), '.png']);
        end
    end
    if sig_neg
        for cl = 1 : length(sig_neg)

            % Indices of the cluster
            idx = stat.negclusterslabelmat == sig_neg(cl);
            pval = round(stat.negclusters(sig_neg(cl)).prob, 3);

            % Save contour of cluster
            dlmwrite([PATH_VEUSZ, titlestring, '_contour_negcluster_', num2str(sig_neg(cl)), '.csv'], idx);
            
            % Identify significant channels and time points
            chans_sig = find(sum(idx, 2));
            times_sig = find(sum(idx, 1));

            % Plot a topo of effect sizes
            markercolor = 'k';
            markersize = 10;
            cmap = 'jet';
            clim = [-0.5, 0.5];
            figure('Visible', 'off'); clf;
            pd = mean(apes(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            saveas(gcf, [PATH_OUTPUT, [titlestring, '_effectsize_topo_neg_'], num2str(sig_neg(cl)), '.png']);

            % Plot
            markercolor = 'k';
            markersize = 10;
            cmap = 'jet';
            clim = [-5, 5];
            figure('Visible', 'off'); clf;

            subplot(2, 2, 1)
            pd = squeeze(mean(cond1.individual, 1));
            contourf(stat.time, [1 : numel(cond1.label)], pd, 40, 'linecolor','none')
            hold on
            contour(stat.time, [1 : numel(cond1.label)], idx, 1, 'linecolor', 'k', 'LineWidth', 2)
            colormap(cmap)
            set(gca, 'clim', [clim])
            colorbar;
            title(cond1string)

            subplot(2, 2, 2)
            pd = squeeze(mean(cond2.individual, 1));
            contourf(stat.time, [1 : numel(cond1.label)], pd, 40, 'linecolor','none')
            hold on
            contour(stat.time, [1 : numel(cond1.label)], idx, 1, 'linecolor', 'k', 'LineWidth', 2)
            colormap(cmap)
            set(gca, 'clim', [clim])
            colorbar;
            title(cond2string)
            
            subplot(2, 2, 3)
            pd = squeeze(mean(cond1.individual, 1));
            pd = mean(pd(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            title(['negcluster #' num2str(sig_neg(cl)) ' - p=' num2str(pval)])

            subplot(2, 2, 4)
            pd = squeeze(mean(cond2.individual, 1));
            pd = mean(pd(:, times_sig), 2);
            topoplot(pd, chanlocs, 'plotrad', 0.7, 'intrad', 0.7, 'intsquare', 'on', 'conv', 'off', 'electrodes', 'off', 'emarker2', {chans_sig, 'p', markercolor, markersize, 1});
            colormap(cmap);
            caxis(clim);
            title(['negcluster #' num2str(sig_neg(cl)) ' - p=' num2str(pval)])

            saveas(gcf, [PATH_OUTPUT, [titlestring, '_neg_'], num2str(sig_neg(cl)), '.png']);
        end
    end
end % End function