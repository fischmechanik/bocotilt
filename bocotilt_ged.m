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

        % participant identifiers
        subject = subject_list{s};
        id = str2num(subject(3 : 4));

        % Load data
        EEG = pop_loadset('filename', [subject_list{s} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

        % To double precision
        eeg_data = double(EEG.data);

        % TODO: Filter-hilbert data 

        % Find indices of time points for S & R selection
		tidx_S = EEG.times >=  100 & EEG.times <=  500;
		tidx_R = EEG.times >= -500 & EEG.times <= -100;
        
        % Init arrays for trial-specific covariance matrices
        covmats_S = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));
        covmats_R = zeros(size(eeg_data, 3), size(eeg_data, 1), size(eeg_data, 1));

        % Covariance matrix for each trial
        for trial_idx = 1 : size(d, 3)

            % Get data for covariance matrices
            data_S = squeeze(eeg_data(:, tidx_S, trial_idx));
            data_R = squeeze(eeg_data(:, tidx_R, trial_idx));

            % Mean center data
            data_S = bsxfun(@minus, data_S, mean(data_S, 2));
            data_R = bsxfun(@minus, data_R, mean(data_R, 2));

            % Compute covariance matrices
            covmats_S(trial_idx, :, :) = data_S * data_S' / (sum(tidx_S) - 1);
            covmats_R(trial_idx, :, :) = data_R * data_R' / (sum(tidx_R) - 1);

        end

        % Compute average covariance matrices
        ave_covmat_S = squeeze(mean(covmats_S, 1));
        ave_covmat_R = squeeze(mean(covmats_R, 1));

        % Init vector for largest eigenvalues from permutation procedure
        extreme_eigs = [];

        % Permute
        for perm = 1 : 1000

            % Draw swappers
            to_swap = randsample(size(covmats_S, 1),  floor(size(covmats_S, 1) / 2));

            % Create permutet covariance matrix collections
            permutet_covmats_S = covmats_S;
            permutet_covmats_S(to_swap, :, :) = squeeze(covmats_R(to_swap, :, :));
            permutet_covmats_R = covmats_R;
            permutet_covmats_R(to_swap, :, :) = squeeze(covmats_S(to_swap, :, :));

            % Average
            ave_permuted_covmat_S = squeeze(mean(permutet_covmats_S, 1));
            ave_permuted_covmat_R = squeeze(mean(permutet_covmats_R, 1));

        
        end

        aa = bb;


        % Prune for event-related analysis
        %tidx = dsearchn(EEG.times', [pruned_segs(1), pruned_segs(2)]');
        %d = d(:, tidx(1) : tidx(2), :);

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
		tidx_S = dsearchn(EEG.times', [300, 700]');
		tidx_R = dsearchn(EEG.times', [300, 700]');

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

        aa = bb;

  

    end

end % End part1