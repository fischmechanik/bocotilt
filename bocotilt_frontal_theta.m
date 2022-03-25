% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2021.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_TFDECOMP    = '/mnt/data_dump/bocotilt/4_tf_decomp/';

% Subject list
subject_list = {'VP08', 'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23'};

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
        focuschan = 'FCz';
        channel_idx = find(strcmp({EEG.chanlocs.labels}, focuschan));
        d = double(squeeze(EEG.data(channel_idx, :, :)));

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
        tmp = fft(reshape(d, 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(d, 1), size(d, 2));
            powcube(f, :, :) = abs(as) .^ 2;          
        end

        % Cut edges
        powcube = powcube(:, dsearchn(EEG.times', prune_times(1)) : dsearchn(EEG.times', prune_times(2)), :);

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
        tf_struct.zcube = zcube;
        save([PATH_TFDECOMP, subject, '_tf_decomp'], 'tf_struct');

    end % End subject iteration

end % End part1

% Part 2: ---
if ismember('part2', to_execute)

    res = [];

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
        tf_struct.zcube(:, :, to_drop) = [];
        tf_struct.powcube(:, :, to_drop) = []; 

        % Get condition idx
        idx_std = tf_struct.trialinfo(:, 4) == 0;
        idx_bon = tf_struct.trialinfo(:, 4) == 1;

        res(s, 1, :, :) = squeeze(mean(tf_struct.zcube(:, :, idx_std), 3));
        res(s, 2, :, :) = squeeze(mean(tf_struct.zcube(:, :, idx_bon), 3));

    end % End subject loop


    figure;
    
    subplot(2, 1, 1)
    pd = squeeze(mean(res(:, 1, :, :), 1));
    contourf(tf_struct.tf_times, tf_struct.tf_freqs, pd, 40, 'linecolor','none')
    colormap('jet')
    set(gca, 'clim', [-22, 22], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    colorbar;
    title(['standard trials'], 'FontSize', 12)

    subplot(2, 1, 2)
    pd = squeeze(mean(res(:, 2, :, :), 1));
    contourf(tf_struct.tf_times, tf_struct.tf_freqs, pd, 40, 'linecolor','none')
    colormap('jet')
    set(gca, 'clim', [-22, 22], 'YScale', 'lin', 'YTick', [4, 8, 12, 20])
    colorbar;
    title(['bonus trials'], 'FontSize', 12)

end % End part2






if false


    % Save component signal and single trial pows and pruned time vector
    save([PATH_COMPONENT_DATA subject '_signal_cmp'], 'cmp_sig');
    save([PATH_COMPONENT_DATA subject '_powcube_cmp'], 'powcube');
    save([PATH_COMPONENT_DATA 'tf_time_cmp'], 'prune_time');

    % Load subject meta
    meta = dlmread([PATH_TFDECOMP num2str(id) '_powcube_meta.csv']);

    % Regression design matrices
    meta_c = meta(meta(:, 7) == 1, :);
    meta_nc = meta(meta(:, 7) == 2, :);
    desmat = [ones(size(meta, 1), 1), meta(:, 5)];
    desmat(:, 2) = desmat(:, 2) / max(abs(desmat(:, 2))); % Scale
    desmat_c = [ones(size(meta_c, 1), 1), meta_c(:, 5)];
    desmat_c(:, 2) = desmat_c(:, 2) / max(abs(desmat_c(:, 2))); % Scale
    desmat_nc = [ones(size(meta_nc, 1), 1), meta_nc(:, 5)];
    desmat_nc(:, 2) = desmat_nc(:, 2) / max(abs(desmat_nc(:, 2))); % Scale

    % Z-Standardize trials
    

    % Apply single trial baseline
    zcube = zeros(size(powcube));
    blidx = dsearchn(prune_time', [-200, 0]');
    for t = 1 : size(powcube, 3)
        d_trial = squeeze(powcube(:, :, t)); % Get trial tfmat
        blvals = squeeze(mean(d_trial(:, blidx(1) : blidx(2)), 2)); % Get baseline
        blstd = std(d_trial(:, blidx(1) : blidx(2)), 0, 2);
        d_trial = bsxfun(@minus, d_trial, blvals);
        zcube(:, :, t) = bsxfun(@rdivide, d_trial, blstd);
    end

    % Set time windows as trial ranges for ersp calculation
    trialwins = {[1 : 200], [1241 : 1440], [2681 : 2880], [4121 : 4320]};

    % Save cmp ersps for timewins
    for w = 1 : numel(trialwins)
        ersp_cmp_c(w, :, :) = squeeze(ersp_cmp_c(w, :, :)) + squeeze(mean(zcube(:, :, ismember(meta(:, 5), trialwins{w}) & meta(:, 7) == 1), 3));
        ersp_cmp_nc(w, :, :) = squeeze(ersp_cmp_nc(w, :, :)) + squeeze(mean(zcube(:, :, ismember(meta(:, 5), trialwins{w}) & meta(:, 7) == 2), 3));
    end

    % Split data
    zcube_c = zcube(:, :, meta(:, 7) == 1);
    zcube_nc = zcube(:, :, meta(:, 7) == 2);

    % OLS fit all trials
    d = reshape(zcube, numel(prune_time) * numel(tf_freqs), size(zcube, 3))';
    tmp = (desmat' * desmat) \ desmat' * d;
    cmp_betas_tru = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
    save([PATH_COMPONENT_DATA subject '_cmp_betas_tru'], 'cmp_betas_tru');

    % Generate tot pseudo condition
    fakedesmat = desmat;
    fakedesmat(:, 2) = desmat(randperm(size(desmat, 1)), 2); % Permute tot column
    tmp = (fakedesmat' * fakedesmat) \ fakedesmat' * d;
    cmp_betas_fak = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
    save([PATH_COMPONENT_DATA subject '_cmp_betas_fak'], 'cmp_betas_fak');

    % OLS fit c trials
    d = reshape(zcube_c, numel(prune_time) * numel(tf_freqs), size(zcube_c, 3))';
    tmp = (desmat_c' * desmat_c) \ desmat_c' * d;
    cmp_betas_tru_c = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
    save([PATH_COMPONENT_DATA subject '_cmp_betas_tru_c'], 'cmp_betas_tru_c');

    % OLS fit nc trials
    d = reshape(zcube_nc, numel(prune_time) * numel(tf_freqs), size(zcube_nc, 3))';
    tmp = (desmat_nc' * desmat_nc) \ desmat_nc' * d;
    cmp_betas_tru_nc = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
    save([PATH_COMPONENT_DATA subject '_cmp_betas_tru_nc'], 'cmp_betas_tru_nc');

    % Save ersp of c and nc trials
    cmp_ersp_c = squeeze(mean(zcube_c, 3));
    cmp_ersp_nc = squeeze(mean(zcube_nc, 3));
    save([PATH_COMPONENT_DATA subject '_cmp_ersp_c'], 'cmp_ersp_c');
    save([PATH_COMPONENT_DATA subject '_cmp_ersp_nc'], 'cmp_ersp_nc');

end % End subit




% ======================= PART8: Generalized eigendecomposition, apply spatial filter, time-frequency analysis and linear regression ==========================================================================

if false

    % Load TF analysis parameters
    pruned_segs = dlmread([PATH_TFDECOMP 'pruned_segs.csv']); % [-2000, 1000]
    tf_times = dlmread([PATH_TFDECOMP 'tf_times.csv']); 

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
    
    % Init ersp resmat for cmcomponentp data
    ersp_cmp_c = zeros(4, length(tf_freqs), length(dsearchn(tf_times', -500) : dsearchn(tf_times', 1000)));
    ersp_cmp_nc = zeros(4, length(tf_freqs), length(dsearchn(tf_times', -500) : dsearchn(tf_times', 1000)));

    % Iterating subject list
    for s = 1 : length(subject_list)

        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load clean data
        EEG = pop_loadset('filename', [subject '_autocleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');
        d = double(EEG.data);

        % Prune for event-related analysis
        tidx = dsearchn(EEG.times', [pruned_segs(1), pruned_segs(2)]');
        d = d(:, tidx(1) : tidx(2), :);

        % Filter data for spatial filter determination
		fwhmf = 3;
		cfrq = 5.5;
		tmp = reshape(d, EEG.nbchan, []); % Trial after trial
		hz = linspace(0, EEG.srate, size(tmp, 2)); % Define frequency vector
		s  = fwhmf * (2 * pi - 1) / (4 * pi); % normalized width
		x  = hz - cfrq; % shifted frequencies
		fx = exp(-.5 * ( x / s) .^ 2); % Create gaussian in frq-domain
		fx = fx ./ max(fx); % normalize gaussian
		tmp = 2 * real(ifft(bsxfun(@times, fft(tmp, [], 2), fx), [], 2)); % Actually filter the data 
		d_filt = reshape(tmp, [EEG.nbchan, size(d, 2), size(d, 3)]); % Back to 3d

		% Find indices of time points for S & R selection
		tidx_S = dsearchn(tf_times', [300, 700]');
		tidx_R = dsearchn(tf_times', [300, 700]');

		% Calculate covariance matrices
		d_S = reshape(d_filt(:, tidx_S(1) : tidx_S(2), :), EEG.nbchan, []);
		d_S = bsxfun(@minus, d_S, mean(d_S, 2)); % Mean center data
		S = d_S * d_S' / diff(tidx_S(1 : 2));

		d_R = reshape(d(:, tidx_R(1) : tidx_R(2), :), EEG.nbchan, []);
		d_R = bsxfun(@minus, d_R, mean(d_R, 2)); % Mean center data
		R = d_R * d_R' / diff(tidx_R(1 : 2));

		% GED and sort eigenvalues and eigenvectors
		[evecs, evals] = eig(S, R);
		[evals, srtidx] = sort(diag(evals), 'descend'); % Sort eigenvalues
		evecs = evecs(:, srtidx); % Sort eigenvectors

		% Normalize eigenvectors
        evecs = bsxfun(@rdivide, evecs, sqrt(sum(evecs .^ 2, 1)));
        
        % Threshold eigenvalues
		thresh_eval_sd = 1; % In sd
		thresh_eigenvalue = median(evals) + std(evals) * thresh_eval_sd;
		suprathresh_eval_idx = find(evals > thresh_eigenvalue);  

        % Iterate components
        ged_maps = zeros(EEG.nbchan, EEG.nbchan);
        ged_tsrs = zeros(EEG.nbchan, length(tf_times), EEG.trials);
		for cmp = 1 : EEG.nbchan

			% Compute maps and flip sign if necessary
			ged_maps(cmp, :) = evecs(:, cmp)' * S;
			[~, idx] = max(abs(ged_maps(cmp, :)));
			ged_maps(cmp, :) = ged_maps(cmp, :) * sign(ged_maps(cmp, idx));

			% Compute time series data for components, i.e. apply spatial filters to unfiltered data
			tpts = evecs(:, cmp)' * reshape(d, EEG.nbchan, []);
			ged_tsrs(cmp, :, :) = reshape(tpts, [length(tf_times), size(d, 3)]);   

        end

        % Identify blink components [Fp1, Fp2, AFz -> 1, 2, 3] > [FC1, FCz, FC2 -> 13, 14, 15]
		blink_cmp = zeros(1, EEG.nbchan);
		for cmp = 1 : EEG.nbchan
			if mean(ged_maps(cmp, [1, 2, 3])) > mean(ged_maps(cmp, [13, 14, 15]))
				blink_cmp(cmp) = 1;
			end
		end

		% Find highest similarity in supra-threshold non-blink cmps
		cmpsim = 0;
		chosen_cmp = 0;
		for e = 1 : EEG.nbchan
			if ismember(e, suprathresh_eval_idx) & ~blink_cmp(e)
				tmp_cmp = ged_maps(e, :) / max(ged_maps(e, :)); % Normalize
				tmp = corrcoef(tmp_cmp, template_topo);
				if tmp(1, 2) * evals(e) > cmpsim
					cmpsim = tmp(1, 2) * evals(e);
					chosen_cmp = e;
				end
			end	
		end

		% Save filter topography
		figure('Visible', 'off'); clf;
		topoplot(ged_maps(chosen_cmp, :), EEG.chanlocs, 'electrodes', 'off', 'numcontour', 0)
        saveas(gcf, [PATH_VEUSZ, 'descriptives_spatial_filter/', 'filter_topo_' subject '.png']);
        
        % Get component signal
        cmp_sig = squeeze(ged_tsrs(chosen_cmp, :, :));

        % tf decomp of component
        convlen = size(cmp_sig, 1) * size(cmp_sig, 2) + size(cmw, 2) - 1;

        % cmw to freq domain and scale
        cmwX = zeros(length(tf_freqs), convlen);
        for f = 1 : length(tf_freqs)
            cmwX(f, :) = fft(cmw(f, :), convlen);
            cmwX(f, :) = cmwX(f, :) ./ max(cmwX(f, :));
        end

        % Get TF-power
        powcube = NaN(length(tf_freqs), size(cmp_sig, 1), size(cmp_sig, 2));
        tmp = fft(reshape(double(cmp_sig), 1, []), convlen);
        for f = 1 : length(tf_freqs)
            as = ifft(cmwX(f, :) .* tmp); 
            as = as(((size(cmw, 2) - 1) / 2) + 1 : end - ((size(cmw, 2) - 1) / 2));
            as = reshape(as, size(cmp_sig, 1), size(cmp_sig, 2));
            powcube(f, :, :) = abs(as) .^ 2;          
        end

        % Cut edges
        powcube = powcube(:, dsearchn(tf_times', -500) : dsearchn(tf_times', 1000), :);
        prune_time = tf_times(dsearchn(tf_times', -500) : dsearchn(tf_times', 1000));

        % Save component signal and single trial pows and pruned time vector
        save([PATH_COMPONENT_DATA subject '_signal_cmp'], 'cmp_sig');
        save([PATH_COMPONENT_DATA subject '_powcube_cmp'], 'powcube');
        save([PATH_COMPONENT_DATA 'tf_time_cmp'], 'prune_time');

        % Load subject meta
        meta = dlmread([PATH_TFDECOMP num2str(id) '_powcube_meta.csv']);

        % Regression design matrices
        meta_c = meta(meta(:, 7) == 1, :);
        meta_nc = meta(meta(:, 7) == 2, :);
        desmat = [ones(size(meta, 1), 1), meta(:, 5)];
        desmat(:, 2) = desmat(:, 2) / max(abs(desmat(:, 2))); % Scale
        desmat_c = [ones(size(meta_c, 1), 1), meta_c(:, 5)];
        desmat_c(:, 2) = desmat_c(:, 2) / max(abs(desmat_c(:, 2))); % Scale
        desmat_nc = [ones(size(meta_nc, 1), 1), meta_nc(:, 5)];
        desmat_nc(:, 2) = desmat_nc(:, 2) / max(abs(desmat_nc(:, 2))); % Scale

        % Z-Standardize trials
        zcube = zeros(size(powcube));

        % Apply single trial baseline
        blidx = dsearchn(prune_time', [-200, 0]');
        for t = 1 : size(powcube, 3)
            d_trial = squeeze(powcube(:, :, t)); % Get trial tfmat
            blvals = squeeze(mean(d_trial(:, blidx(1) : blidx(2)), 2)); % Get baseline
            blstd = std(d_trial(:, blidx(1) : blidx(2)), 0, 2);
            d_trial = bsxfun(@minus, d_trial, blvals);
            zcube(:, :, t) = bsxfun(@rdivide, d_trial, blstd);
        end

        % Set time windows as trial ranges for ersp calculation
        trialwins = {[1 : 200], [1241 : 1440], [2681 : 2880], [4121 : 4320]};

        % Save cmp ersps for timewins
        for w = 1 : numel(trialwins)
            ersp_cmp_c(w, :, :) = squeeze(ersp_cmp_c(w, :, :)) + squeeze(mean(zcube(:, :, ismember(meta(:, 5), trialwins{w}) & meta(:, 7) == 1), 3));
            ersp_cmp_nc(w, :, :) = squeeze(ersp_cmp_nc(w, :, :)) + squeeze(mean(zcube(:, :, ismember(meta(:, 5), trialwins{w}) & meta(:, 7) == 2), 3));
        end

        % Split data
        zcube_c = zcube(:, :, meta(:, 7) == 1);
        zcube_nc = zcube(:, :, meta(:, 7) == 2);

        % OLS fit all trials
        d = reshape(zcube, numel(prune_time) * numel(tf_freqs), size(zcube, 3))';
        tmp = (desmat' * desmat) \ desmat' * d;
        cmp_betas_tru = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
        save([PATH_COMPONENT_DATA subject '_cmp_betas_tru'], 'cmp_betas_tru');

        % Generate tot pseudo condition
        fakedesmat = desmat;
        fakedesmat(:, 2) = desmat(randperm(size(desmat, 1)), 2); % Permute tot column
        tmp = (fakedesmat' * fakedesmat) \ fakedesmat' * d;
        cmp_betas_fak = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
        save([PATH_COMPONENT_DATA subject '_cmp_betas_fak'], 'cmp_betas_fak');

        % OLS fit c trials
        d = reshape(zcube_c, numel(prune_time) * numel(tf_freqs), size(zcube_c, 3))';
        tmp = (desmat_c' * desmat_c) \ desmat_c' * d;
        cmp_betas_tru_c = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
        save([PATH_COMPONENT_DATA subject '_cmp_betas_tru_c'], 'cmp_betas_tru_c');

        % OLS fit nc trials
        d = reshape(zcube_nc, numel(prune_time) * numel(tf_freqs), size(zcube_nc, 3))';
        tmp = (desmat_nc' * desmat_nc) \ desmat_nc' * d;
        cmp_betas_tru_nc = reshape(squeeze(tmp(2, :)), [numel(tf_freqs), numel(prune_time)]);
        save([PATH_COMPONENT_DATA subject '_cmp_betas_tru_nc'], 'cmp_betas_tru_nc');

        % Save ersp of c and nc trials
        cmp_ersp_c = squeeze(mean(zcube_c, 3));
        cmp_ersp_nc = squeeze(mean(zcube_nc, 3));
        save([PATH_COMPONENT_DATA subject '_cmp_ersp_c'], 'cmp_ersp_c');
        save([PATH_COMPONENT_DATA subject '_cmp_ersp_nc'], 'cmp_ersp_nc');

    end % End subit

    % Average and save timewin cmp ersps
    for w = 1 : numel(trialwins)
        ersp_cmp_c(w, :, :) = squeeze(ersp_cmp_c(w, :, :)) / numel(subject_list);
        dlmwrite([PATH_VEUSZ, 'descriptives_spatial_filter/', 'ersp_cmp_c_win_' num2str(w) '.csv'], squeeze(ersp_cmp_c(w, :, :)));

        ersp_cmp_nc(w, :, :) = squeeze(ersp_cmp_nc(w, :, :)) / numel(subject_list);
        dlmwrite([PATH_VEUSZ, 'descriptives_spatial_filter/', 'ersp_cmp_nc_win_' num2str(w) '.csv'], squeeze(ersp_cmp_nc(w, :, :)));
    end

    % Load  data
    d_tru = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    d_fak = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    d_tru_c = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    d_tru_nc = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    d_ersp_c = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    d_ersp_nc = zeros(length(subject_list), length(tf_freqs), length(prune_time));
    for s = 1 : length(subject_list)
        subject = subject_list{s};
        load([PATH_COMPONENT_DATA subject '_cmp_betas_tru']);
        load([PATH_COMPONENT_DATA subject '_cmp_betas_fak']);
        d_tru(s, :, :) = cmp_betas_tru;
        d_fak(s, :, :) = cmp_betas_fak;
        load([PATH_COMPONENT_DATA subject '_cmp_betas_tru_c']);
        load([PATH_COMPONENT_DATA subject '_cmp_betas_tru_nc']);
        d_tru_nc(s, :, :) = cmp_betas_tru_c;
        d_fak_nc(s, :, :) = cmp_betas_tru_nc;
        load([PATH_COMPONENT_DATA subject '_cmp_ersp_c']);
        load([PATH_COMPONENT_DATA subject '_cmp_ersp_nc']);
        d_ersp_c(s, :, :) = cmp_ersp_c;
        d_ersp_nc(s, :, :) = cmp_ersp_nc;
    end

    % Save average data
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tot_beta.csv'], squeeze(mean(d_tru, 1)));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tot_beta_c.csv'], squeeze(mean(d_tru_c, 1)));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tot_beta_nc.csv'], squeeze(mean(d_tru_nc, 1)));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tt_ersp_c.csv'], squeeze(mean(d_ersp_c, 1)));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tt_ersp_nc.csv'], squeeze(mean(d_ersp_nc, 1)));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tt_ersp_diff.csv'], squeeze(mean(d_ersp_nc, 1)) - squeeze(mean(d_ersp_c, 1)));

    % Permtest params
    pval_voxel   = 0.01;
    pval_cluster = 0.025;
    n_perms      = 1000;

    % Test against zero c data
    d1 = d_tru;
    d2 = d_fak;
    n_freq = length(tf_freqs);
    n_time = length(prune_time);
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
    contourres = logical(tmat);

    % Save contour of effect
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tot_contour.csv'], contourres);

    % Calculate and save effect sizes
    petasq = (tvals .^ 2) ./ ((tvals .^ 2) + (numel(subject_list) - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (numel(subject_list) - 1));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tot_effect_sizes.csv'], adj_petasq);

    % Test tt
    d1 = d_ersp_c;
    d2 = d_ersp_nc;
    n_freq = length(tf_freqs);
    n_time = length(prune_time);
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
    contourres = logical(tmat);

    % Save contour of effect
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tt_contour.csv'], contourres);

    % Calculate and save effect sizes
    petasq = (tvals .^ 2) ./ ((tvals .^ 2) + (numel(subject_list) - 1));
    adj_petasq = petasq - (1 - petasq) .* (1 / (numel(subject_list) - 1));
    dlmwrite([PATH_VEUSZ, 'inference_spatial_filter/', 'tt_effect_sizes.csv'], adj_petasq);

end % End part8

% ======================= PART9: SUBJECTIVE RATINGS =====================================

if ismember('part9', to_execute)

    % Read data
    sr = xlsread([PATH_RAW_DATA 'subjective_ratings.xlsx']);

    % Option to leave out first assessments after the breaks and at start
    if false
        sr(ismember(sr(:, 2), [1, 5, 9]), :) = [];
        sr(sr(:, 2) == 2, 2) = 1;
        sr(sr(:, 2) == 3, 2) = 2;  
        sr(sr(:, 2) == 4, 2) = 3;  
        sr(sr(:, 2) == 6, 2) = 4;  
        sr(sr(:, 2) == 7, 2) = 5;  
        sr(sr(:, 2) == 8, 2) = 6;  
        sr(sr(:, 2) == 10, 2) = 7;  
        sr(sr(:, 2) == 11, 2) = 8;  
        sr(sr(:, 2) == 12, 2) = 9;  
    end

    % Flip polarity of motivation
    sr(:, 4) = (sr(:, 4) * -1) + 10;

    % Create tables
    varnames = {'id', 'tot' , 'fat', 'mot'};
    tbl = table(sr(:, 1), sr(:, 2), sr(:, 3), sr(:, 4), 'VariableNames', varnames);
    
    % Cast vars
    tbl.id = nominal(tbl.id);

    % Compute LMEs
    lme_fat = fitlme(tbl, 'fat ~ tot + (1|id)');
    lme_mot = fitlme(tbl, 'mot ~ tot + (1|id)');

    % Calculate averages
    out = [];
    ts = unique(sr(:, 2));
    for t = 1 : length(ts)
        tmp = sr(sr(:, 2) == ts(t), 3);
        fat_m = mean(tmp);
        fat_s = std(tmp);
        tmp = sr(sr(:, 2) == ts(t), 4);
        mot_m = mean(tmp);
        mot_s = std(tmp);
        out(t, :) = [fat_m, fat_s, mot_m, mot_s];
    end

    dlmwrite([PATH_VEUSZ, 'descriptives_subjective/', 'ratings.csv'], out, 'delimiter', '\t');
    dlmwrite([PATH_VEUSZ, 'descriptives_subjective/', 'xax.csv'], [1 : length(ts)]);

end % End part9