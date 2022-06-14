% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.0/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_GED         = '/mnt/data_dump/bocotilt/7_ged/';
PATH_VEUSZ       = '/mnt/data_dump/bocotilt/veusz/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31'};


%subject_list = {'VP09', 'VP17', 'VP25', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part3'};

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

        % Exclude trials
        to_keep = EEG.trialinfo(:, 2) > 4 & EEG.trialinfo(:, 17) == 1 & EEG.trialinfo(:, 22) > 1 & EEG.trialinfo(:, 22) < 8;
        eeg_data = eeg_data(:, :, to_keep);
        EEG.trialinfo = EEG.trialinfo(to_keep, :);
        EEG.trials = sum(to_keep);

        % Binarize block info
        block_binarized = zeros(EEG.trials, 1);
        block_binarized(EEG.trialinfo(:, 2) > 7) = 1;
        EEG.trialinfo = [EEG.trialinfo, block_binarized];

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
		tidx_S = (times >= 100 & times <= 700);
		tidx_R = (times >= 100 & times <= 700);
        
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

        % Init vector for largest eigenvalues from permutation procedure
        extreme_eigs = [];

        % Permute
        n_perm = 10;
        for perm = 1 : n_perm

            fprintf('Creating 0-hypothesis distribution - step %i/%i\n', perm, n_perm);

            % Draw swappers
            min_n = min([size(covmats_S, 1), size(covmats_R, 1)]);
            to_swap = randsample(min_n,  floor(min_n / 2));

            % Create permutet covariance matrix collections
            permutet_covmats_S = covmats_S;
            permutet_covmats_S(to_swap, :, :) = squeeze(covmats_R(to_swap, :, :));
            permutet_covmats_R = covmats_R;
            permutet_covmats_R(to_swap, :, :) = squeeze(covmats_S(to_swap, :, :));

            % Average
            ave_permuted_covmat_S = squeeze(mean(permutet_covmats_S, 1));
            ave_permuted_covmat_R = squeeze(mean(permutet_covmats_R, 1));

            % Apply shrinkage regularization
            g = 0.1;
            a = mean(eig(ave_permuted_covmat_R));
            ave_permuted_covmat_R = (1 - g) * ave_permuted_covmat_R + g * a * eye(EEG.nbchan);

            % GED 
            [~, evals_permuted] = eig(ave_permuted_covmat_S, ave_permuted_covmat_R);

            % Get maximum eigenvalue
            [maxeig, ~] = max(diag(evals_permuted));

            % Save largest eigenvalue
            extreme_eigs(end + 1) = maxeig;
            
        end

        % Sort
        [extreme_eigs, ~] = sort(extreme_eigs);

        % Determine threshold (looking for eigs >= thresh...)
        thresh = extreme_eigs(ceil(n_perm / 100 * 5));

        % Iterate components
        n_comps = EEG.nbchan;
        ged_maps = zeros(n_comps, EEG.nbchan);
        ged_time_series = zeros(n_comps, length(EEG.times), EEG.trials);
        ged_time_series_theta = zeros(n_comps, length(times), EEG.trials);
        comp_sigs = zeros(n_comps, 1);
        for cmp = 1 : n_comps

            % Get significance
            if evals(cmp) >= thresh
                comp_sigs(cmp) = 1;
            end

            % Compute map
            ged_maps(cmp, :) = evecs(:, cmp)' * S;

            % Flip map if necessary
            [~, idx] = max(abs(ged_maps(cmp, :)));
            ged_maps(cmp, :) = ged_maps(cmp, :) * sign(ged_maps(cmp, idx));

            % Compute time series for component
            component_time_series = evecs(:, cmp)' * eeg_data_2d;

            % Get a filtered version (theta) of time series
            component_time_series_theta = filtfilt(filter_weights, 1, component_time_series);

            % Reshape time seriesto 3d
            ged_time_series(cmp, :, :) = reshape(component_time_series, [length(EEG.times), EEG.trials]);

            % Compute power for filtered data
            component_time_series_theta = abs(hilbert(component_time_series_theta')') .^ 2;

            % Reshape theta time seriesto 3d
            tmp = reshape(component_time_series_theta, [length(EEG.times), EEG.trials]);

            % Crop in time
            ged_time_series_theta(cmp, :, :) = tmp(crop_idx, :);

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
        % 24: block_binarized

        % Getting some nice indices
        idx_std_rep = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 0;
        idx_std_swi = EEG.trialinfo(:, 4) == 0 & EEG.trialinfo(:, 10) == 1;
        idx_bon_rep = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 0;
        idx_bon_swi = EEG.trialinfo(:, 4) == 1 & EEG.trialinfo(:, 10) == 1;

        % Some viz
        figure('Visible', 'off'); clf;
        n_cols = 3;
        n_rows = 8;
        for cmp = 1 : n_rows

            % Is significant?
            if comp_sigs(cmp)
                sigchar = '*';
            else
                sigchar = '';
            end

            % Find figure-index offset
            fig_idx_offset = (cmp - 1) * n_cols;

            % Plot topo
            subplot(n_rows, n_cols, fig_idx_offset + 1)
            pd = ged_maps(cmp, :);
            topoplot(pd, EEG.chanlocs, 'electrodes', 'off', 'numcontour', 0)
            title(['eig-val: ', num2str(evals(cmp)), sigchar]);

            % Plot component time series
            subplot(n_rows, n_cols, fig_idx_offset + 2)
            pd = mean(squeeze(ged_time_series(cmp, :, :)), 2);
            plot(EEG.times, pd, 'LineWidth', 1.5)
            xline(0)

            % Plot component theta time series
            subplot(n_rows, n_cols, fig_idx_offset + 3)
            pd_std_rep = mean(squeeze(ged_time_series_theta(cmp, :, idx_std_rep)), 2);
            pd_std_swi = mean(squeeze(ged_time_series_theta(cmp, :, idx_std_swi)), 2);
            pd_bon_rep = mean(squeeze(ged_time_series_theta(cmp, :, idx_bon_rep)), 2);
            pd_bon_swi = mean(squeeze(ged_time_series_theta(cmp, :, idx_bon_swi)), 2);

            plot(times, pd_std_rep, 'LineWidth', 1.5)
            hold on
            plot(times, pd_std_swi, 'LineWidth', 1.5)
            plot(times, pd_bon_rep, 'LineWidth', 1.5)
            plot(times, pd_bon_swi, 'LineWidth', 1.5)
            xline(0)
        end

        % Save figure
        saveas(gcf, [PATH_GED, 'plot_ged_', subject, '.png']);

        % Save stuff
        result = struct();
        result.extreme_eigs = extreme_eigs;
        result.eig_thresh = thresh;
        result.trialinfo = EEG.trialinfo;
        result.ged_time_series = single(ged_time_series);
        result.ged_maps = ged_maps;
        result.eeg_data = single(eeg_data);
        result.times = EEG.times;
        result.evecs = evecs;
        result.evals = evals;
        result.S = S;
        result.R = R;
        save([PATH_GED, 'ged_result_', subject, '.mat'], 'result');

    end

end % End part1


% Part 2: tf-decomp
if ismember('part2', to_execute)

    % Load info
    EEG = pop_loadset('filename', [subject_list{1} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'info');

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

    % Plot spatial filter map template
    figure('Visible', 'off'); clf;
    topoplot(template_topo, EEG.chanlocs, 'electrodes', 'off', 'numcontour', 0)
    title(['template topo'])
    set(gcf, 'PaperUnits', 'centimeters')
    set(gcf, 'PaperPosition', [0, 0, 10, 10])
    saveas(gcf, [PATH_GED, 'spatial_filter_template.png']);
    
    % Set complex Morlet wavelet parameters
    n_frq = 50;
    frqrange = [2, 20];
    tfres_range = [400, 150];

    % Create wavelet frequencies and tapering Gaussian widths in temporal domain
    tf_freqs = linspace(frqrange(1), frqrange(2), n_frq);
    fwhmTs = logspace(log10(tfres_range(1)), log10(tfres_range(2)), n_frq);

    % Create wavelets
    wtime = -2 : 1 / EEG.srate : 2;
    for frq = 1 : length(tf_freqs)
        cmw(frq, :) = exp(2 * 1i * pi * tf_freqs(frq) .* wtime) .* exp((-4 * log(2) * wtime.^2) ./ (fwhmTs(frq) / 1000)^2);
        cmw(frq, :) = cmw(frq, :) ./ max(cmw(frq, :));
    end

    % Init a result struct
    tf_result = struct();
    tf_result.template_topo = template_topo;
    tf_result.tf_freqs = tf_freqs;
    tf_result.fwhmTs = fwhmTs;
    tf_result.ersp = {};
    tf_result.subjects = {};

    % Loop subjects
    for s = 1 : length(subject_list)

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));
        tf_result.subjects{end + 1} = subject;

        % Load data
        load([PATH_GED, 'ged_result_', subject, '.mat']);

        % Threshold eigenvalues
		thresh_eval_sd = 1; % In sd
		thresh_eigenvalue = median(result.evals) + std(result.evals) * thresh_eval_sd;
		suprathresh_eval_idx = find(result.evals > thresh_eigenvalue);  

        % Find highest similarity in supra-threshold non-blink cmps
		cmpsim = 0;
		chosen_cmp = 0;
		for cmp = 1 : EEG.nbchan
			if ismember(cmp, suprathresh_eval_idx)
				tmp_cmp = result.ged_maps(cmp, :) / max(result.ged_maps(cmp, :)); % Normalize
				%tmp = abs(corrcoef(tmp_cmp, template_topo));
                tmp = corrcoef(tmp_cmp, template_topo);
				if tmp(1, 2) * result.evals(cmp) > cmpsim
					cmpsim = tmp(1, 2) * result.evals(cmp);
					chosen_cmp = cmp;
				end
			end	
		end

        % Save filter topography
		figure('Visible', 'off'); clf;
		topoplot(result.ged_maps(chosen_cmp, :), EEG.chanlocs, 'electrodes', 'off', 'numcontour', 0)
        saveas(gcf, [PATH_GED, 'spatial_filter_topo_' subject '.png']);
        tf_result.cmp_topo = result.ged_maps(chosen_cmp, :);

        % Get component signal
        cmp_time_series = squeeze(result.ged_time_series(chosen_cmp, :, :));

        % tf decomp of component
        convlen = size(cmp_time_series, 1) * size(cmp_time_series, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(length(tf_freqs), convlen);
        for f = 1 : length(tf_freqs)
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        powcube = NaN(length(tf_freqs), size(cmp_time_series, 1), size(cmp_time_series, 2));
        tmp = fft(reshape(double(cmp_time_series), 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(cmp_time_series, 1), size(cmp_time_series, 2));
            powcube(f, :, :) = abs(as) .^ 2; 
        end

        % Cut edges
        powcube = powcube(:, dsearchn(result.times', -500) : dsearchn(result.times', 1600), :);
        prune_time = result.times(dsearchn(result.times', -500) : dsearchn(result.times', 1600));
        tf_result.tf_time = prune_time;

        % Get condition general baseline values
        bl_idx = prune_time >= -500 & prune_time <= -200;
        bl_vals = squeeze(mean(powcube, 3));

        % Calculate averages for conditions
        ersp = zeros(2, 2, size(powcube, 1), size(powcube, 2));
        for bon = 1 : 2
            for swi = 1 : 2

                % Get trial idx
                trial_idx = find(result.trialinfo(:, 4) == bon - 1 & result.trialinfo(:, 10) == swi - 1);

                % Get condition mean of power
                condition_mean_power = squeeze(mean(powcube(:, :, trial_idx), 3));

                % Calculate ersp
                ersp(bon, swi, :, :) = 10 * log10(bsxfun(@rdivide, condition_mean_power, bl_vals));
                
            end
        end

        % Copy to results
        tf_result.ersp{end + 1} = ersp;

    end % End subject loop

    % Save tf results
    save([PATH_GED, 'tf_results.mat'], 'tf_result');

end % End part2



% Part 3: Analysis
if ismember('part3', to_execute)

    % Load tf results
    load([PATH_GED, 'tf_results.mat']);

    % Concatenate tf data. Dims: bonus, switch, freqs, times, subjects
    ersps = cat(5, tf_result.ersp{:});

    % Get condition data
    ersp_std_rep = permute(squeeze(ersps(1, 1, :, :, :)), [3, 1, 2]);
    ersp_std_swi = permute(squeeze(ersps(1, 2, :, :, :)), [3, 1, 2]);
    ersp_bon_rep = permute(squeeze(ersps(2, 1, :, :, :)), [3, 1, 2]);
    ersp_bon_swi = permute(squeeze(ersps(2, 2, :, :, :)), [3, 1, 2]);

    % Matrix for theta traces
    theta_ersp_traces = [];

    % Plot ersp and calculate theta traces
    bonus_labels = {'std', 'bonus'};
    switch_labels = {'repeat', 'switch'};
    figure()
    counter = 0;

    for bon = 1 : 2
        for swi = 1 : 2

            counter = counter + 1;

            % Plot data
            pd_ersp = squeeze(mean(squeeze(ersps(bon, swi, :, :, :)), 3));


            subplot(2, 2, counter)
            contourf(tf_result.tf_time, tf_result.tf_freqs, pd_ersp, 40, 'linecolor','none')
            colormap('jet')
            set(gca, 'clim', [-1, 1], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])

            titlestring = ['bon: ', num2str(bon), ' - swi: ', num2str(swi)];
            title(titlestring, 'FontSize', 10);

            % get theta_trace
            theta_idx = tf_result.tf_freqs >= 4 & tf_result.tf_freqs <= 8;
            theta_ersp_traces(counter, :) = mean(pd_ersp(theta_idx, :), 1);

        end
    end

    % Main effect bonus
    data1 = (ersp_std_rep + ersp_std_swi) / 2;
    data2 = (ersp_bon_rep + ersp_bon_swi) / 2;
    main_effect_bonus = struct();
    [main_effect_bonus.sig_flag,...
        main_effect_bonus.ave1,...
        main_effect_bonus.ave2,...
        main_effect_bonus.outline,...
        main_effect_bonus.apes,...
        main_effect_bonus.clust_sumt,...
        main_effect_bonus.clust_pvals,...
        main_effect_bonus.clust_apes,...
        main_effect_bonus.time_limits,...
        main_effect_bonus.freq_limits,...
        main_effect_bonus.cluster_idx] = cluststats_2d_data(data1, data2, tf_result.tf_time, tf_result.tf_freqs);

    figure()
    contourf(tf_result.tf_time, tf_result.tf_freqs, main_effect_bonus.apes, 50, 'linecolor','none')
    hold on
    contour(tf_result.tf_time, tf_result.tf_freqs, main_effect_bonus.outline, 1, 'linecolor', 'k', 'LineWidth', 2)
    colorbar
    colormap('jet')
    set(gca, 'clim', [-0.5, 0.5], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    title('Main effect bonus')

    % Main effect switch
    data1 = (ersp_std_rep + ersp_bon_rep) / 2;
    data2 = (ersp_std_swi + ersp_bon_swi) / 2;
    main_effect_switch = struct();
    [main_effect_switch.sig_flag,...
        main_effect_switch.ave1,...
        main_effect_switch.ave2,...
        main_effect_switch.outline,...
        main_effect_switch.apes,...
        main_effect_switch.clust_sumt,...
        main_effect_switch.clust_pvals,...
        main_effect_switch.clust_apes,...
        main_effect_switch.time_limits,...
        main_effect_switch.freq_limits,...
        main_effect_switch.cluster_idx] = cluststats_2d_data(data1, data2, tf_result.tf_time, tf_result.tf_freqs);

    figure()
    contourf(tf_result.tf_time, tf_result.tf_freqs, main_effect_switch.apes, 50, 'linecolor','none')
    hold on
    contour(tf_result.tf_time, tf_result.tf_freqs, main_effect_switch.outline, 1, 'linecolor', 'k', 'LineWidth', 2)
    colorbar
    colormap('jet')
    set(gca, 'clim', [-0.5, 0.5], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    title('Main effect switch')

    % Interaction
    data1 = ersp_std_rep - ersp_std_swi;
    data2 = ersp_bon_rep - ersp_bon_swi;
    interaction_effect = struct();
    [interaction_effect.sig_flag,...
        interaction_effect.ave1,...
        interaction_effect.ave2,...
        interaction_effect.outline,...
        interaction_effect.apes,...
        interaction_effect.clust_sumt,...
        interaction_effect.clust_pvals,...
        interaction_effect.clust_apes,...
        interaction_effect.time_limits,...
        interaction_effect.freq_limits,...
        interaction_effect.cluster_idx] = cluststats_2d_data(data1, data2, tf_result.tf_time, tf_result.tf_freqs);

    figure()
    contourf(tf_result.tf_time, tf_result.tf_freqs, interaction_effect.apes, 50, 'linecolor','none')
    hold on
    contour(tf_result.tf_time, tf_result.tf_freqs, interaction_effect.outline, 1, 'linecolor', 'k', 'LineWidth', 2)
    colorbar
    colormap('jet')
    set(gca, 'clim', [-0.5, 0.5], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    title('interaction')

    % Save for plots
    dlmwrite([PATH_VEUSZ, 'ersp_std_rep.csv'], squeeze(mean(ersp_std_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'ersp_std_swi.csv'], squeeze(mean(ersp_std_swi, 1)));
    dlmwrite([PATH_VEUSZ, 'ersp_bon_rep.csv'], squeeze(mean(ersp_bon_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'ersp_bon_swi.csv'], squeeze(mean(ersp_bon_swi, 1)));
    dlmwrite([PATH_VEUSZ, 'diff_swi_rep_in_std.csv'], squeeze(mean(ersp_std_swi - ersp_std_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'diff_swi_rep_in_bon.csv'], squeeze(mean(ersp_bon_swi - ersp_bon_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'diff_bon_std_in_rep.csv'], squeeze(mean(ersp_bon_rep - ersp_std_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'diff_bon_std_in_swi.csv'], squeeze(mean(ersp_bon_swi - ersp_std_swi, 1)));
    dlmwrite([PATH_VEUSZ, 'diff_of_diff.csv'], squeeze(mean(ersp_bon_swi - ersp_bon_rep, 1)) - squeeze(mean(ersp_std_swi - ersp_std_rep, 1)));
    dlmwrite([PATH_VEUSZ, 'apes_std_vs_bon.csv'], main_effect_bonus.apes);
    dlmwrite([PATH_VEUSZ, 'apes_rep_vs_swi.csv'], main_effect_switch.apes);
    dlmwrite([PATH_VEUSZ, 'apes_interaction.csv'], interaction_effect.apes);
    dlmwrite([PATH_VEUSZ, 'bonus_contour.csv'], main_effect_bonus.outline);
    dlmwrite([PATH_VEUSZ, 'switch_contour.csv'], main_effect_switch.outline);
    dlmwrite([PATH_VEUSZ, 'interaction_contour.csv'], interaction_effect.outline);



    aa=bb 


    % Save theta ersp traces
    dlmwrite([PATH_GED, 'theta_ersp_traces.csv'], theta_ersp_traces);
    dlmwrite([PATH_GED, 'theta_traces_times.csv'], tf_result.tf_time);

    % Calculate the grand-average theta-time-series
    grand_average_theta_ersp = mean(theta_ersp_traces, 1);

    % Pop up plot of peaks in the theta grand-average
    %figure()
    %findpeaks(grand_average_theta_ersp)

    %figure()
    %findpeaks(grand_average_theta_itpc)

    % Save peak indices.
    [peakamps_ersp, theta_peak_indices_ersp] = findpeaks(grand_average_theta_ersp);
    [peakamps_itpc, theta_peak_indices_itpc] = findpeaks(grand_average_theta_itpc, 'MinPeakDistance', 10);

    % Reduce itpc peaks to large amplitudes
    theta_peak_indices_itpc = theta_peak_indices_itpc(peakamps_itpc > 0.15);

    % Define width of the ersp time window 
    winwidth_ersp = 100;

    % Determine time-windows around peaks
    theta_window_ersp_1 = [tf_result.tf_time(theta_peak_indices_ersp(1)) - winwidth_ersp / 2, tf_result.tf_time(theta_peak_indices_ersp(1)) + winwidth_ersp / 2];
    theta_window_ersp_2 = [tf_result.tf_time(theta_peak_indices_ersp(2)) - winwidth_ersp / 2, tf_result.tf_time(theta_peak_indices_ersp(2)) + winwidth_ersp / 2]; 

    % Define width of the itpc time window 
    winwidth_itpc = 100;

    % Determine time-windows around peaks
    theta_window_itpc_1 = [tf_result.tf_time(theta_peak_indices_itpc(1)) - winwidth_itpc / 2, tf_result.tf_time(theta_peak_indices_itpc(1)) + winwidth_itpc / 2];
    theta_window_itpc_2 = [tf_result.tf_time(theta_peak_indices_itpc(2)) - winwidth_itpc / 2, tf_result.tf_time(theta_peak_indices_itpc(2)) + winwidth_itpc / 2]; 

    % Iterate subjects and conditions again and parameterize theta
    theta_table = [];
    counter = 0;
    for s = 1 : length(subject_list)
        for tot = 1 : 2
            for bon = 1 : 2
                for swi = 1 : 2
        
                    counter = counter + 1;

                    % Get time idx for ersp
                    time_idx_win_1 = tf_result.tf_time >= theta_window_ersp_1(1) & tf_result.tf_time <= theta_window_ersp_1(2);
                    time_idx_win_2 = tf_result.tf_time >= theta_window_ersp_2(1) & tf_result.tf_time <= theta_window_ersp_2(2);

                    % get averages for ersp
                    th1 = mean2(ersps(tot, bon, swi, theta_idx, time_idx_win_1, s));
                    th2 = mean2(ersps(tot, bon, swi, theta_idx, time_idx_win_2, s));

                    % Get time idx for itpc
                    time_idx_win_1 = tf_result.tf_time >= theta_window_itpc_1(1) & tf_result.tf_time <= theta_window_itpc_1(2);
                    time_idx_win_2 = tf_result.tf_time >= theta_window_itpc_2(1) & tf_result.tf_time <= theta_window_itpc_2(2);

                    % get averages for itpc
                    th3 = mean2(itpcs(tot, bon, swi, theta_idx, time_idx_win_1, s));
                    th4 = mean2(itpcs(tot, bon, swi, theta_idx, time_idx_win_2, s));

                    % Collect
                    theta_table(counter, :) = [s, tot, bon, swi, th1, th2, th3, th4, tf_result.rts(tot, bon, swi), tf_result.accuracies(tot, bon, swi)];

                end
            end
        end
    end

    save([PATH_GED, 'theta_table.mat'], 'theta_table');
    


end % End part3