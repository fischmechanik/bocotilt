% Clear residuals
clear all;

% Path variables
PATH_EEGLAB      = '/home/plkn/eeglab/';
PATH_AUTOCLEANED = '/mnt/data2/bocotilt/2_autocleaned/';

% Subject list
subject_list = {'VP09', 'VP17', 'VP25', 'VP10', 'VP11', 'VP13', 'VP14', 'VP15', 'VP16', 'VP18',...
                'VP19', 'VP20', 'VP21', 'VP22', 'VP23', 'VP08', 'VP24', 'VP26', 'VP27', 'VP28',...
                'VP29', 'VP30', 'VP31', 'VP32', 'VP33', 'VP34'};

% Init eeglab
addpath(PATH_EEGLAB);
eeglab;

% Load some data
EEG = pop_loadset('filename', [subject_list{1} '_cleaned.set'], 'filepath', PATH_AUTOCLEANED, 'loadmode', 'all');

% Get data as channel x time x epochs
eeg_data = double(EEG.data);

% Get dimession
[n_channel, n_time, n_epoch] = size(eeg_data);

% Get sampling rate
fs = EEG.srate;

% Parameters for SFFT windows
win_size     = 200;     % window size in frames
win_stepsize = 10;      % window stepsize in frames
win_nave     = 5;       % number of windows being averaged per PSD

% Make number of time points in window even
win_size = win_size - mod(win_size, 2);

% Determine time window start idx
win_start_idx = 1 : win_stepsize : n_time - win_size;

% Number of windows
n_win = length(win_start_idx);

% Determine time window center time
win_center_times = EEG.times(win_start_idx + round(win_size / 2));

% Length of fft
nfft = 256;

% Positive frequencies of FFT
fft_freqs = fs / 2 * linspace(0, 1, nfft / 2 + 1);

% Determine hann window shape and power
hann_win = hann(win_size);
hann_win_noise_power_gain = hann_win' * hann_win;

% Repeat window for each trial
hann_win = repmat(hann_win, 1, n_epoch);

% Spectrogram matrix (channel x timewin x freqs x trials)
spectrograms = NaN(n_channel, n_win, length(fft_freqs), n_epoch);

% Iterate channels
for ch = 1 : n_channel

    % Iterate time windows
    for w = 1 : n_win

        % Get
        win_idx = win_start_idx(w) : win_start_idx(w) + win_size - 1;

        % Select window data (time x epoch)
        win_data = squeeze(eeg_data(ch, win_idx, :));

        % No need to enforce removing DC component (0 frequency).
        win_data = win_data - mean(win_data, 2);

        % Apply a Hann window to signal
        win_data = win_data .* hann_win;

        % Compute FFT
        win_data_fft = fft(win_data, nfft, 1);

        % Normalize
        win_data_fft = win_data_fft / hann_win_noise_power_gain;

        % Get first half of power spectrum
        win_data_fft = win_data_fft(1 : nfft / 2 + 1, :);
        
        % Multiply by 2 (except DC & Nyquist) 
        win_data_fft(2 : end - 1, :) = win_data_fft(2 : end - 1, :) * 2;

        % Divide by fs and square for power
        win_data_fft = (win_data_fft / fs);

        % Convert to power
        win_data_fft = abs(win_data_fft) .^ 2;

        % Save
        spectrograms(ch, w, :, :) = win_data_fft;
    
    end
end

spectrograms = squeeze(mean(spectrograms(127, :, :, :), 4));


figure()
plot(fft_freqs, squeeze(mean(spectrograms(2, :, :), 4)))
contourf(fft_freqs, win_center_times, spectrograms)


aa=bb

    % Initialize STFT, time matrices
    ts = nan(Nwin-(avgWin-1),1);
    TF = nan(size(F,1), Nwin-(avgWin-1), size(FreqVector,2));
    TFtmp = nan(size(F,1), avgWin, size(FreqVector,2));

    % ===== CALCULATE FFT FOR EACH WINDOW =====
    TFfull = zeros(size(F, 1), Nwin, size(FreqVector, 2));
    for iWin = 1 : Nwin

        % Build indices
        iTimes = (1:Lwin) + (iWin-1)*(Lwin - Loverlap);
        center_time = floor(median((iTimes-(avgWin-1)./2*(Lwin - Loverlap))))./sfreq;

        % Select indices
        Fwin = F(:, iTimes);

        % No need to enforce removing DC component (0 frequency).
        Fwin = Fwin - mean(Fwin, 2);

        % Apply a Hann window to signal
        Fwin = Fwin .* Win;

        % Compute FFT
        Ffft = fft(Fwin, NFFT, 2);

        % One-sided spectrum (keep only first half)
        % (x2 to recover full power from negative frequencies)
        TFwin = Ffft(:, 1 : NFFT / 2 + 1) * sqrt(2 ./ (sfreq * WinNoisePowerGain));

        % x2 doesn't apply to DC and Nyquist.
        TFwin(:, [1, end]) = TFwin(:, [1, end]) ./ sqrt(2);

        % Permute dimensions: time and frequency
        TFwin = permute(TFwin, [1 3 2]);

        % Convert to power
        TFwin = abs(TFwin).^2;
        TFfull(:, iWin, :) = TFwin;
        TFtmp(:, mod(iWin, avgWin) + 1, :) = TFwin;

        if isnan(TFtmp(1, 1, 1)) 
            continue % Do not record anything until transient is gone
        else
            % Save STFTs for window
            TF(:, iWin - (avgWin - 1), :) = mean(TFtmp, 2);
            ts(iWin - (avgWin - 1)) = center_time;
        end
    end

    % From here on TF is expected to be 3 dimensional (channels x times x freqs)


