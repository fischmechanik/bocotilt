% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.0/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18', 'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% SWITCH: Switch parts of script on/off
to_execute = {'part1'};

% Part 1: tf-decomp
if ismember('part1', to_execute)

    % Loop subjects
    for s = 1 : length(subject_list)

        % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        s = 4;

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

        % Construct filter
        nyquist = EEG.srate / 2;
        h_pass = 4;
        l_pass = 8;
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
        crop_idx = EEG.times >= -500 & EEG.times <= 2000;
        times = EEG.times(crop_idx);
        eeg_data_filtered = eeg_data_filtered(:, crop_idx, :);

        % Find indices of time points for S & R selection
		tidx_S = times >= 100 & times <= 800;
		tidx_R = times >= 100 & times <= 800;
        
        % Init arrays for trial-specific covariance matrices
        covmats_S = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));
        covmats_R = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));

        % Covariance matrix for each trial
        for trial_idx = 1 : size(eeg_data, 3)

            % Get data for covariance matrices
            data_S = squeeze(eeg_data_filtered(:, tidx_S, trial_idx));
            data_R = squeeze(eeg_data_filtered(:, tidx_R, trial_idx));

            % Mean center data
            data_S = bsxfun(@minus, data_S, mean(data_S, 2));
            data_R = bsxfun(@minus, data_R, mean(data_R, 2));

            % Compute covariance matrices
            covmats_S(trial_idx, :, :) = data_S * data_S' / (sum(tidx_S) - 1);
            covmats_R(trial_idx, :, :) = data_R * data_R' / (sum(tidx_R) - 1);

        end

        % Compute covariance matrices for switch and repeat trials
        idx_switch = find(EEG.trialinfo(:, 10) == 1 & EEG.trialinfo(:, 17) == 1);
        idx_repeat = find(EEG.trialinfo(:, 10) == 0 & EEG.trialinfo(:, 17) == 1);
        covmats_S_switch = zeros(length(idx_switch), size(eeg_data, 1), size(eeg_data, 1));
        covmats_R_repeat = zeros(length(idx_repeat), size(eeg_data, 1), size(eeg_data, 1));
        for trial_idx = 1 : length(idx_switch)

            % Get data 
            tmp = squeeze(eeg_data_filtered(:, tidx_S, idx_switch(trial_idx)));

            % Mean center data
            tmp = bsxfun(@minus, tmp, mean(tmp, 2));

            % Compute covariance matrices
            covmats_S_switch(trial_idx, :, :) = tmp * tmp' / (sum(tidx_S) - 1);

        end
        for trial_idx = 1 : length(idx_repeat)

            % Get data 
            tmp = squeeze(eeg_data_filtered(:, tidx_S, idx_repeat(trial_idx)));

            % Mean center data
            tmp = bsxfun(@minus, tmp, mean(tmp, 2));

            % Compute covariance matrices
            covmats_R_repeat(trial_idx, :, :) = tmp * tmp' / (sum(tidx_S) - 1);

        end

        % Select covariance matrices to actually use... (!)
        S_single_trial = covmats_S_switch;
        R_single_trial = covmats_R_repeat;

        % Compute average covariance matrices
        S = squeeze(mean(S_single_trial, 1));
        R = squeeze(mean(R_single_trial, 1));

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
        n_perm = 100;
        for perm = 1 : n_perm

            fprintf('Creating 0-hypothesis distribution - step %i/%i\n', perm, n_perm);

            % Draw swappers
            min_n = min([size(S_single_trial, 1), size(R_single_trial, 1)]);
            to_swap = randsample(min_n,  floor(min_n / 2));

            % Create permutet covariance matrix collections
            permutet_covmats_S = S_single_trial;
            permutet_covmats_S(to_swap, :, :) = squeeze(R_single_trial(to_swap, :, :));
            permutet_covmats_R = R_single_trial;
            permutet_covmats_R(to_swap, :, :) = squeeze(S_single_trial(to_swap, :, :));

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
        thresh = extreme_eigs(n_perm / 100 * 5);

        % Iterate components
        n_comps = 8;
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

        % Getting some nice indices
        idx_standard = EEG.trialinfo(:, 4) == 1; 
        idx_bonus = EEG.trialinfo(:, 4) == 0; 

        % Some viz
        figure()
        n_cols = 3;
        n_rows = n_comps;
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
            xline([0, 800])

            % Plot component theta time series
            subplot(n_rows, n_cols, fig_idx_offset + 3)
            pd_standard = mean(squeeze(ged_time_series_theta(cmp, :, idx_standard)), 2);
            pd_bonus = mean(squeeze(ged_time_series_theta(cmp, :, idx_bonus)), 2);
            plot(times, pd_standard, 'LineWidth', 1.5)
            hold on
            plot(times, pd_bonus, 'LineWidth', 1.5)
            xline([0, 800])
            legend('std', 'bonus')
        end

        aa = bb;







    end

end % End part1