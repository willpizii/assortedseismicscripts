import numpy as np

def symmetrise_trace(disp, fudges = None):

    trace = disp.trace_raw.copy()

    d = trace.data.astype(float)
    d /= np.max(np.abs(d))

    if not fudges:
        mid = d.size // 2
        d = d[:2*mid]
        s = 0.5 * (d[mid:] + d[:mid][::-1])
        
    else:       
        fudge = 0

        if get_attribute(disp.sta1,'sta') in fudges.keys():
            fudge += fudges[get_attribute(disp.sta1,'sta')]
        if get_attribute(disp.sta2,'sta') in fudges.keys():
            fudge -= fudges[get_attribute(disp.sta2,'sta')]

        if fudge != 0:
            shift_samples = int(fudge * trace.stats.sampling_rate)
            mid = d.size // 2
            new_mid = mid + shift_samples

            left = d[new_mid:]
            right = d[:new_mid][::-1]

            min_len = min(len(left), len(right))
            s = 0.5 * (left[:min_len] + right[:min_len])

        else:
            mid = d.size // 2
            d = d[:2*mid]
            s = 0.5 * (d[mid:] + d[:mid][::-1])   

    trace.data = s
    return trace

def get_attribute(s, k):

    fun = lambda s, k: getattr(s, k, None) or s[k]
    return fun(s,k)

def signal_time_to_velocity(disp, trace, mode, centre_period = None):

    t = trace.times()
    
    if mode == 'phase':
        if centre_period is None:
            raise ValueError("centre period required for phase correction")
        
        vmask = t > centre_period / 8
        v = disp.dist / (t[vmask]- centre_period / 8) # Yao et al 2006, phase correction
    else:
        vmask = t > 0
        v = disp.dist / t[vmask]

    return v, vmask

def get_signal_snr(disp, trace, mode, **kwargs):

        v, vmask = signal_time_to_velocity(disp, trace, mode, **kwargs)
        data = trace.data[vmask]

        # Noise: RMS of the trace outside the zone of interest
        signal_mask = (v >= disp.minV) & (v <= disp.maxV)
        noise_mask  = (v < disp.minV / disp.snr_pad)

        if mode == 'group':
            # For envelope/CWT: peak signal vs RMS noise
            peak_signal = np.max(data[signal_mask])
            rms_noise = np.sqrt(np.mean(data[noise_mask]**2))
            return peak_signal / (rms_noise + 1e-12)
        else:
            # For phase: RMS signal vs RMS noise
            rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
            rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
            return rms_signal / (rms_noise + 1e-12)

def get_wavelengths(distance, centre_period, velocity):
    return distance / (centre_period * velocity)