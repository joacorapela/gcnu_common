import numpy as np

def compute_acf(spikes_times):
    """Computes autocorrelations functions for all trials and mean autocorrelation function across trials

    :param spikes_times: ``spikes_times[r]`` contains binned spikes times from trial `r`. Bins should be 1~ms in length.
    :type  spikes_times: numpy array with trials along the rows and bins across the columns

    :returns autocorrelation functions of individual trials, mean autocorrelation function across trials and lags
    :rtype numpy array of shape (n_trials, n_bins), numpy array of length n_trials and numpy array of length n_trials
    """
    n_trials = spikes_times.shape[0]
    n_bins = spikes_times.shape[1]
    acf = np.empty((n_trials, 2*n_bins-1), dtype=np.double)
    for r in range(n_trials):
        trial_spikes_times = spikes_times[r, :]
        x = trial_spikes_times - np.mean(trial_spikes_times)
        corr = np.correlate(x, x, "full")
        acf[r, :] = corr / np.linalg.norm(x)**2
    acf = acf[:, n_bins:(2*n_bins-1)]
    lags_ms = np.arange(n_bins)
    mean_acf = np.mean(acf, axis=0)

    return acf, mean_acf, lags_ms
def compute_spectrum(x, Fs):
    dt = 1 / Fs                      # Define the time step
    T = len(x) * dt                  # Define the total time
    X = np.fft.rfft(x - x.mean())    # Compute the Fourier transform
    Sxx = np.real((X * np.conj(X)))  # ... and the spectrum
    norm = Sxx.max()
    if norm > 0:
        Sxx = Sxx / norm             # ... and scale it to have maximum of 0.

    df = 1 / T                       # Define the frequency resolution,
    faxis = np.arange(len(Sxx)) * df # ... to create frequency axis
    return Sxx, faxis


def compute_mean_spectrum(x_trials, Fs):
    n_trials = x_trials.shape[0]
    n_bins = x_trials.shape[1]
    n_freq_points = int((n_bins)/2)+1 if n_bins%2==0 else int((n_bins+1)/2)
    neuron_spectrums = np.empty((n_trials, n_freq_points), dtype=np.double)
    for r in range(n_trials):
        trial_spectrum, freqs = compute_spectrum(x=x_trials[r,:],
                                                 Fs=Fs)
        neuron_spectrums[r,:] = trial_spectrum
    mean_trial_spectrum = np.mean(neuron_spectrums, axis=0)
    return mean_trial_spectrum, freqs


def compute_spectogram(t, x_trials, window_length, step_size, Fs, fpass):
    window_length_samples, step_size_samples = \
            [int(Fs * x) for x in [window_length, step_size]]    # Convert step and window to samples.
    starts = range(0, x_trials.shape[-1] - window_length_samples,
                   step_size_samples)    # Determine where spectrogram windows should start.
    f = compute_mean_spectrum(
        x_trials=x_trials[:, range(window_length_samples)], Fs=Fs)[1] # Get the frequencies,
    findx = (f >= fpass[0]) & (f <= fpass[1])                    # ... create a mask of frequencies of interest,
    f = f[findx]                                                 # ... and select these frequencies of interest.
    spectogram = []
    for s in starts:                                             # Compute the spectrum on each 500 ms window.
        spectrum = compute_mean_spectrum(                        # ...  starting every 50 ms
            x_trials[:, range(s, s + window_length_samples)],
            Fs=Fs)[0][findx]
        spectogram.append(spectrum)

    spectogram = np.array(spectogram).T
    T = t[starts] + window_length_samples / 2                                   # Centers of spectrogram windows.
    return spectogram, T, f
