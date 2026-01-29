import numpy as np

# As described by Pavlis and Vernon, 2010
# Shown by Yang et al. 2023 to be good for phase dispersion and other analysis

def snr_stack(ncfs, distance:float, sample_rate:float, bandpass = None, velocity_range = [1000,4000]):
    """
    Takes given signals, such as ncfs, and returns the snr-based stack
    
    Parameters:
    ncfs: np.stack of traces to process; in stack of N traces x M samples
    distance: inter-station distance
    sample_rate: sampling rate of the signal
    bandpass: None (no filter) or [minfreq,maxfreq] bandpass frequencies (not implemented)
    velocity_range: [minvel, maxvel] velocity (in m/s) range of interest. Defaults to [1000,4000]

    Returns:
    b: final weighted snr stack
    weights: weights of each input data trace
    """

    # cast input trace stack to array
    ncfs = np.asarray(ncfs, dtype=float)

    # normalise amplitudes of input traces
    ncfs /= np.linalg.norm(ncfs, axis=1, keepdims=True)

    # take shape - N number of traces, M trace length
    _, M = ncfs.shape

    mid = (M - 1) // 2

    snr_dict = {}

    for j, x in enumerate(ncfs):
        x1 = np.flip(x[0:mid])
        x2 = x[mid+1:]

        sym = x1 + x2

        t = np.arange(0,len(sym) * 1 / sample_rate, 1/sample_rate) 

        v = distance / t
        data = sym

        # Noise: RMS of the entire trace
        signal_mask = (v >= velocity_range[0]) & (v <= velocity_range[1])
        noise_mask  = (v < velocity_range[0]) | (v > velocity_range[1])

        rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
        rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
        snr = rms_signal / (rms_noise + 1e-12)

        snr_dict[j] = snr
    
    sum_snr = np.sum([v for v in snr_dict.values()])
    snr_dict = {k: v/sum_snr for k,v in snr_dict.items()}

    weights = np.array([v for v in snr_dict.values()])
    b = np.sum(weights[:, None] * ncfs, axis=0)

    return b, weights