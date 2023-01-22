% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/home/plkn/bocotilt_fooof/cleaned/';
PATH_GED         = '/home/plkn/bocotilt_fooof/ged_time_series/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};


% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part1'};

% Part 1: Calculate ged
if ismember('part1', to_execute)

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

        % 01: id
        % 02: block_nr
        % 03: trial_nr
        % 04: bonustrial
        % 05: tilt_task
        % 06: cue_ax
        % 07: target_red_left
        % 08: distractor_red_left
        % 09: response_interference
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
        to_keep = EEG.trialinfo(:, 2) > 4 & EEG.trialinfo(:, 23) > 1;
        eeg_data = eeg_data(:, :, to_keep);
        EEG.trialinfo = EEG.trialinfo(to_keep, :);
        EEG.trials = sum(to_keep);

        % Save FCz
        fcz_time_series = squeeze(eeg_data(127, :, :));

        % Construct filter
        nyquist = EEG.srate / 2;
        h_pass = 4;
        l_pass = 7;
        transition_width = 0.2;
        filter_order = round(3 * (EEG.srate / h_pass));
        filter_freqs = [0, (1 - transition_width) * h_pass, h_pass, l_pass, (1 + transition_width) * l_pass, nyquist] / nyquist; 
        filter_response = [0, 0, 1, 1, 0, 0];
        filter_weights = firls(filter_order, filter_freqs, filter_response);

        % Reshape to 2d
        eeg_data_2d = reshape(eeg_data, [EEG.nbchan, EEG.pnts * EEG.trials]);

        % Apply filter
        eeg_data_filtered_2d = zeros(size(eeg_data_2d));
        for ch = 1 : EEG.nbchan
            eeg_data_filtered_2d(ch, :) = filtfilt(filter_weights, 1, eeg_data(ch, :));
        end

        % Reshape back to 3d
        eeg_data_filtered = reshape(eeg_data_filtered_2d, [EEG.nbchan, EEG.pnts, EEG.trials]);

        % Crop in time to remove edge artifacts
        crop_idx = EEG.times >= -500 & EEG.times <= 1600;
        times = EEG.times(crop_idx);
        eeg_data_filtered = eeg_data_filtered(:, crop_idx, :);

        % Find indices of time points for S & R selection
		tidx_S = (times >= 300 & times <= 800);
		tidx_R = (times >= 300 & times <= 800);
        
        % Init arrays for trial-specific covariance matrices
        covmats_S = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));
        covmats_R = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));

        % Covariance matrix for each trial
        for trial_idx = 1 : size(eeg_data, 3)

            % Get data for covariance matrices
            data_S = squeeze(eeg_data_filtered(:, tidx_S, trial_idx));
            data_R = squeeze(eeg_data(:, tidx_R, trial_idx));

            % Mean center data
            data_S = bsxfun(@minus, data_S, mean(data_S, 2));
            data_R = bsxfun(@minus, data_R, mean(data_R, 2));

            % Compute covariance matrices
            covmats_S(trial_idx, :, :) = data_S * data_S' / (sum(tidx_S) - 1);
            covmats_R(trial_idx, :, :) = data_R * data_R' / (sum(tidx_R) - 1);

        end

        % Compute average covariance matrices
        S = squeeze(mean(covmats_S, 1));
        R = squeeze(mean(covmats_R, 1));

        % Apply shrinkage regularization to reference matrices
        g = 0.1;
        a = mean(eig(R));
        R = (1 - g) * R + g * a * eye(EEG.nbchan);

        % GED 
		[evecs, evals] = eig(S, R);

        % Sort eigenvalues and eigenvectors
		[evals, srtidx] = sort(diag(evals), 'descend');
		evecs = evecs(:, srtidx);

        % Normalize eigenvectors
        evecs = bsxfun(@rdivide, evecs, sqrt(sum(evecs .^ 2, 1)));

        % Iterate components
        n_comps = EEG.nbchan;
        ged_maps = zeros(n_comps, EEG.nbchan);
        ged_time_series = zeros(n_comps, length(EEG.times), EEG.trials);

        for cmp = 1 : n_comps

            % Compute map
            ged_maps(cmp, :) = evecs(:, cmp)' * S;

            % Flip map if necessary
            [~, idx] = max(abs(ged_maps(cmp, :)));
            ged_maps(cmp, :) = ged_maps(cmp, :) * sign(ged_maps(cmp, idx));

            % Compute time series for component
            component_time_series = evecs(:, cmp)' * eeg_data_2d;

            % Reshape time series to 3d
            ged_time_series(cmp, :, :) = reshape(component_time_series, [length(EEG.times), EEG.trials]);

        end

        % Determine electrode distances based on cartesian coordinates (loosely adopted on elec_distance.m)
        dists = zeros(EEG.nbchan);
        cart_coords = [cell2mat({EEG.chanlocs.X})', cell2mat({EEG.chanlocs.Y})', cell2mat({EEG.chanlocs.Z})'];
        for ch1 = 1 : EEG.nbchan
            crds1 = cart_coords(ch1, :);
            len1 = sqrt(sum(crds1 .^ 2));
            for ch2 = 1 : EEG.nbchan
                crds2 = cart_coords(ch2, :);
                len2 = sqrt(sum(crds2 .^ 2));
                if ch1 == ch2
                    dists(ch1, ch2) = 0;
                else
                    r = (len1 + len2) / 2; % Estimate sphere radius from len1 and len2
                    theta = acos(dot(crds1, crds2) / (len1 * len2)); % Angle between A & B in radians
                    dists(ch1, ch2) = r * theta; % Arc length = radius * theta
                end
            end
        end

        % Create spatial filter template
        focuschan = 127;
        template_topo = dists(focuschan, :) / max(dists(focuschan, :)); % Normalize distances
        template_topo = ones(size(template_topo)) - template_topo; % Invert

        % Threshold eigenvalues
        thresh_eval_sd = 1; % In sd
        thresh_eigenvalue = median(evals) + std(evals) * thresh_eval_sd;
        suprathresh_eval_idx = find(evals > thresh_eigenvalue);  

        % Find highest similarity in supra-threshold non-blink cmps
        cmpsim = 0;
        chosen_cmp = 0;
        for cmp = 1 : EEG.nbchan
            if ismember(cmp, suprathresh_eval_idx)
                tmp_cmp = ged_maps(cmp, :) / max(ged_maps(cmp, :)); % Normalize
                %tmp = abs(corrcoef(tmp_cmp, template_topo));
                tmp = corrcoef(tmp_cmp, template_topo);
                if tmp(1, 2) * evals(cmp) > cmpsim
                    cmpsim = tmp(1, 2) * evals(cmp);
                    chosen_cmp = cmp;
                end
            end	
        end

        % Get selected component topography
        cmp_topo = ged_maps(chosen_cmp, :);

        % Save filter topography
        figure('Visible', 'off'); clf;
        topoplot(cmp_topo, EEG.chanlocs, 'electrodes', 'off', 'numcontour', 0)
        saveas(gcf, [PATH_GED, 'component_topo_plots/', 'spatial_filter_topo_' subject '.png']);
        
        % Get selected component signal
        cmp_time_series = ged_time_series(chosen_cmp, :, :);

        % Isolate some info
        trialinfo = EEG.trialinfo;
        times = EEG.times;

        % Save selected component activation as mat file
        save([PATH_GED, 'component_time_series/', subject, '_ged_component.mat'], 'cmp_time_series', 'fcz_time_series', 'trialinfo', 'times')

    end % End subject iteration

end % End part1

% Part 2: tf-analysis
if ismember('part2', to_execute)

    % Set complex Morlet wavelet parameters
    srate = 200;
    n_frq = 20;
    frqrange = [2, 20];
    tfres_range = [500, 200];

    % Set wavelet time
    wtime = -2 : 1 / srate : 2;

    % Determine fft frqs
    hz = linspace(0, srate, length(wtime));

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
    load([PATH_GED, 'component_time_series/', subject_list{1}, '_ged_component.mat']);
    tf_times = times(dsearchn(times', prune_times(1)) : dsearchn(times', prune_times(2)));

    % Result matrices
    ersp_std_rep = single(zeros(length(subject_list), length(tf_freqs), length(tf_times)));
    ersp_std_swi = single(zeros(length(subject_list), length(tf_freqs), length(tf_times)));
    ersp_bon_rep = single(zeros(length(subject_list), length(tf_freqs), length(tf_times)));
    ersp_bon_swi = single(zeros(length(subject_list), length(tf_freqs), length(tf_times)));

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Talk
        fprintf('\ntf-decompose subject %i/%i\n', s, length(subject_list));

        % Load data
        load([PATH_GED, 'component_time_series/', subject, '_ged_component.mat']);

        % Squeeze
        cmp_data = squeeze(cmp_time_series);

        % Get condition idx
        idx_std_rep = trialinfo(:, 4) == 0 & trialinfo(:, 10) == 0;
        idx_std_swi = trialinfo(:, 4) == 0 & trialinfo(:, 10) == 1;
        idx_bon_rep = trialinfo(:, 4) == 1 & trialinfo(:, 10) == 0;
        idx_bon_swi = trialinfo(:, 4) == 1 & trialinfo(:, 10) == 1;

        % Init tf matrices
        powcube = NaN(length(tf_freqs), size(cmp_data, 1), size(cmp_data, 2));

        % convolution length
        convlen = size(cmp_data, 1) * size(cmp_data, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(length(tf_freqs), convlen);
        for f = 1 : length(tf_freqs)
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        tmp = fft(reshape(cmp_data, 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(cmp_data, 1), size(cmp_data, 2));
            powcube(f, :, :) = abs(as) .^ 2;   
        end
        
        % Cut edges
        powcube = powcube(:, dsearchn(times', -500) : dsearchn(times', 2000), :);

        % Get condition general baseline values
        ersp_bl = [-500, -200];
        tmp = squeeze(mean(powcube, 3));
        [~, blidx1] = min(abs(tf_times - ersp_bl(1)));
        [~, blidx2] = min(abs(tf_times - ersp_bl(2)));
        blvals = squeeze(mean(tmp(:, blidx1 : blidx2), 2));

        % Calculate ersp
        ersp_std_rep(s, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_rep), 3)), blvals)));
        ersp_std_swi(s, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_std_swi), 3)), blvals)));
        ersp_bon_rep(s, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_rep), 3)), blvals)));
        ersp_bon_swi(s, :, :) = single(10 * log10(bsxfun(@rdivide, squeeze(mean(powcube(:, :, idx_bon_swi), 3)), blvals)));

    end % End subject iteration

    % Permtest params
    n_perms = 1000;
    pval_voxel = 0.01;
    pval_cluster = 0.05;

    % Test reward
    d1 = (ersp_std_rep + ersp_std_swi) / 2;
    d2 = (ersp_bon_rep + ersp_bon_swi) / 2;
    n_freq = length(tf_freqs);
    n_time = length(tf_times);
    permuted_t = zeros(n_perms, n_freq, n_time);
    max_clust = zeros(n_perms, 2);
    desmat = [zeros(length(subject_list), 1), ones(length(subject_list), 1)];
    for p = 1 : n_perms
        fprintf('%i\n', p);
        toflip = find(round(rand(length(subject_list), 1)));
        d1_perm = d1;
        d1_perm(toflip, :, :) = d2(toflip, :, :);
        d2_perm = d2;
        d2_perm(toflip, :, :) = d1(toflip, :, :);
        tnum = squeeze(mean(d1_perm - d2_perm, 1));
        tdenum = squeeze(std(d1_perm - d2_perm, 0, 1)) / sqrt(length(subject_list));
        fake_t = tnum ./ tdenum;
        permuted_t(p, :, :) = fake_t;
        fake_t(abs(fake_t) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
        clusts = bwconncomp(fake_t);
        sum_t = [];
        for clu = 1 : numel(clusts.PixelIdxList)
            cidx = clusts.PixelIdxList{clu};
            sum_t(end + 1) = sum(fake_t(cidx));
        end
        max_clust(p, 1) = min([0, sum_t]);
        max_clust(p, 2) = max([0, sum_t]);      
    end
    tnum = squeeze(mean(d1 - d2, 1));
    tdenum = squeeze(std(d1 - d2, 0, 1)) / sqrt(length(subject_list));
    tmat = tnum ./ tdenum;
    tvals = tmat;
    tmat(abs(tmat) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
    threshtvals = tmat;
    clusts = bwconncomp(tmat);
    sum_t = [];
    for clu = 1 : numel(clusts.PixelIdxList)
        cidx = clusts.PixelIdxList{clu};
        sum_t(end + 1) = sum(tmat(cidx));
    end
    clust_thresh_lower = prctile(max_clust(:, 1), pval_cluster * 100);
    clust_thresh_upper = prctile(max_clust(:, 2), 100 - pval_cluster * 100);
    clust2remove = find(sum_t > clust_thresh_lower & sum_t < clust_thresh_upper);
    for clu = 1 : length(clust2remove)
        tmat(clusts.PixelIdxList{clust2remove(clu)}) = 0;
    end
    contour_reward = logical(tmat);

    % Test switch
    d1 = (ersp_std_rep + ersp_bon_rep) / 2;
    d2 = (ersp_std_swi + ersp_bon_swi) / 2;
    n_freq = length(tf_freqs);
    n_time = length(tf_times);
    permuted_t = zeros(n_perms, n_freq, n_time);
    max_clust = zeros(n_perms, 2);
    desmat = [zeros(length(subject_list), 1), ones(length(subject_list), 1)];
    for p = 1 : n_perms
        fprintf('%i\n', p);
        toflip = find(round(rand(length(subject_list), 1)));
        d1_perm = d1;
        d1_perm(toflip, :, :) = d2(toflip, :, :);
        d2_perm = d2;
        d2_perm(toflip, :, :) = d1(toflip, :, :);
        tnum = squeeze(mean(d1_perm - d2_perm, 1));
        tdenum = squeeze(std(d1_perm - d2_perm, 0, 1)) / sqrt(length(subject_list));
        fake_t = tnum ./ tdenum;
        permuted_t(p, :, :) = fake_t;
        fake_t(abs(fake_t) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
        clusts = bwconncomp(fake_t);
        sum_t = [];
        for clu = 1 : numel(clusts.PixelIdxList)
            cidx = clusts.PixelIdxList{clu};
            sum_t(end + 1) = sum(fake_t(cidx));
        end
        max_clust(p, 1) = min([0, sum_t]);
        max_clust(p, 2) = max([0, sum_t]);      
    end
    tnum = squeeze(mean(d1 - d2, 1));
    tdenum = squeeze(std(d1 - d2, 0, 1)) / sqrt(length(subject_list));
    tmat = tnum ./ tdenum;
    tvals = tmat;
    tmat(abs(tmat) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
    threshtvals = tmat;
    clusts = bwconncomp(tmat);
    sum_t = [];
    for clu = 1 : numel(clusts.PixelIdxList)
        cidx = clusts.PixelIdxList{clu};
        sum_t(end + 1) = sum(tmat(cidx));
    end
    clust_thresh_lower = prctile(max_clust(:, 1), pval_cluster * 100);
    clust_thresh_upper = prctile(max_clust(:, 2), 100 - pval_cluster * 100);
    clust2remove = find(sum_t > clust_thresh_lower & sum_t < clust_thresh_upper);
    for clu = 1 : length(clust2remove)
        tmat(clusts.PixelIdxList{clust2remove(clu)}) = 0;
    end
    contour_switch = logical(tmat);

    % Test interaction
    d1 = (ersp_std_swi - ersp_std_rep);
    d2 = (ersp_bon_swi - ersp_bon_rep);
    n_freq = length(tf_freqs);
    n_time = length(tf_times);
    permuted_t = zeros(n_perms, n_freq, n_time);
    max_clust = zeros(n_perms, 2);
    desmat = [zeros(length(subject_list), 1), ones(length(subject_list), 1)];
    for p = 1 : n_perms
        fprintf('%i\n', p);
        toflip = find(round(rand(length(subject_list), 1)));
        d1_perm = d1;
        d1_perm(toflip, :, :) = d2(toflip, :, :);
        d2_perm = d2;
        d2_perm(toflip, :, :) = d1(toflip, :, :);
        tnum = squeeze(mean(d1_perm - d2_perm, 1));
        tdenum = squeeze(std(d1_perm - d2_perm, 0, 1)) / sqrt(length(subject_list));
        fake_t = tnum ./ tdenum;
        permuted_t(p, :, :) = fake_t;
        fake_t(abs(fake_t) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
        clusts = bwconncomp(fake_t);
        sum_t = [];
        for clu = 1 : numel(clusts.PixelIdxList)
            cidx = clusts.PixelIdxList{clu};
            sum_t(end + 1) = sum(fake_t(cidx));
        end
        max_clust(p, 1) = min([0, sum_t]);
        max_clust(p, 2) = max([0, sum_t]);      
    end
    tnum = squeeze(mean(d1 - d2, 1));
    tdenum = squeeze(std(d1 - d2, 0, 1)) / sqrt(length(subject_list));
    tmat = tnum ./ tdenum;
    tvals = tmat;
    tmat(abs(tmat) < tinv(1 - pval_voxel, length(subject_list) - 1)) = 0;
    threshtvals = tmat;
    clusts = bwconncomp(tmat);
    sum_t = [];
    for clu = 1 : numel(clusts.PixelIdxList)
        cidx = clusts.PixelIdxList{clu};
        sum_t(end + 1) = sum(tmat(cidx));
    end
    clust_thresh_lower = prctile(max_clust(:, 1), pval_cluster * 100);
    clust_thresh_upper = prctile(max_clust(:, 2), 100 - pval_cluster * 100);
    clust2remove = find(sum_t > clust_thresh_lower & sum_t < clust_thresh_upper);
    for clu = 1 : length(clust2remove)
        tmat(clusts.PixelIdxList{clust2remove(clu)}) = 0;
    end
    contour_interaction = logical(tmat);

    % Plots
    figure()

    subplot(2, 2, 1)
    pd = squeeze(mean(ersp_std_rep, 1));
    contourf(tf_times, tf_freqs, pd, 50, 'linecolor','none')
    hold on
    contour(tf_times, tf_freqs, contour_interaction, 1, 'linecolor', 'k', 'LineWidth', 2)
    colormap('jet')
    set(gca, 'clim', [-2, 2], 'YTick', [4, 8, 12, 20, 30])
    colorbar;
    title('std-rep')

    subplot(2, 2, 2)
    pd = squeeze(mean(ersp_std_swi, 1));
    contourf(tf_times, tf_freqs, pd, 50, 'linecolor','none')
    hold on
    contour(tf_times, tf_freqs, contour_switch, 1, 'linecolor', 'k', 'LineWidth', 2)
    colormap('jet')
    set(gca, 'clim', [-2, 2], 'YTick', [4, 8, 12, 20, 30])
    colorbar;
    title('std-swi')

    subplot(2, 2, 3)
    pd = squeeze(mean(ersp_bon_rep, 1));
    contourf(tf_times, tf_freqs, pd, 50, 'linecolor','none')
    hold on
    contour(tf_times, tf_freqs, contour_reward, 1, 'linecolor', 'k', 'LineWidth', 2)
    colormap('jet')
    set(gca, 'clim', [-2, 2], 'YTick', [4, 8, 12, 20, 30])
    colorbar;
    title('bon-rep')

    subplot(2, 2, 4)
    pd = squeeze(mean(ersp_bon_swi, 1));
    contourf(tf_times, tf_freqs, pd, 50, 'linecolor','none')
    hold on
    contour(tf_times, tf_freqs, contour_reward, 1, 'linecolor', 'k', 'LineWidth', 2)
    colormap('jet')
    set(gca, 'clim', [-2, 2], 'YTick', [4, 8, 12, 20, 30])
    colorbar;
    title('bon-swi')

end % End part2