% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab2022.1/';
PATH_AUTOCLEANED = '/mnt/data_dump/bocotilt/2_autocleaned/';
PATH_GED         = '/mnt/data_dump/bocotilt/8_ged77/';

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
        save([PATH_GED, 'component_time_series/', subject, '_ged_component.mat'], 'cmp_time_series', 'trialinfo', 'times')

    end % End subject iteration

end % End part1
