import obspy, os
import numpy as np
from scipy import signal

class ftan:

    def __init__(self,
                 sta1,
                 sta2,
                 cc_file:str,
                 **kwargs):
        """_summary_

        Args:
            sta1 (_type_): station object
            sta2 (_type_): station object
            cc_file (str): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        self.dist, _, _ = obspy.geodetics.base.gps2dist_azimuth(sta1.Y,sta1.X,sta2.Y,sta2.X)
    
        self.sta1 = sta1
        self.sta2 = sta2

        if not cc_file or not os.path.exists(cc_file):
            raise ValueError("CC file required")
        
        self.trace_raw = obspy.read(cc_file)[0]
        self.trace = self._symmetrise_trace(**kwargs)

        del self.trace_raw

    def configure(self,
                  spacing_type:str = 'snr',
                  filter_type:str = 'butterworth',
                  minP:float = 0.5,
                  maxP:float = 10.0,
                  minV:float = 1000,
                  maxV:float = 4500,
                  dP:float = None,
                  snr_cutoff:float = 5.0,
                  snr_pad:float = 2.0,
                  overlap:float = 0.0,
                  wavelength_cutoff:float = 3.0):
        """
        _summary_

        Args:
            spacing_type (str, optional): _description_. Defaults to 'snr'.
            filter_type (str, optional): _description_. Defaults to 'butterworth'.
            minP (float, optional): _description_. Defaults to 0.5.
            maxP (float, optional): _description_. Defaults to 10.0.
            minV (float, optional): _description_. Defaults to 1000.
            maxV (float, optional): _description_. Defaults to 4500.
            dP (float, optional): _description_. Defaults to None.
            snr_cutoff (float, optional): _description_. Defaults to 5.0.
            snr_pad (float, optional): _description_. Defaults to 2.0.
            overlap (float, optional): _description_. Defaults to 0.0.
            wavelength_cutoff (float, optional): _description_. Defaults to 3.0.
        """

        self.minP = minP
        self.maxP = maxP
        self.minV = minV
        self.maxV = maxV
        self.spacing_type = spacing_type
        self.filter_type = filter_type
        self.snr_cutoff = snr_cutoff
        self.snr_pad = snr_pad
        self.wavelength_cutoff = wavelength_cutoff

        if spacing_type == 'snr':
            self.centres, self.fbounds = self._get_snr_filters()

        elif spacing_type == 'fixed_period':
            self.dP = dP if dP else 0.25
            self.centres = np.arange(minP, maxP + dP, dP)

            factor = dP / (2.0 - overlap)
            self.fbounds = [[1.0 / max(c - factor, 1e-6),
                            1.0 / (c + factor)]
                              for c in self.centres]
        
        elif spacing_type == 'logarithmic':
            self.dP = dP if dP else 0.05
            self.centres = np.logspace(np.log10(minP), np.log10(maxP),
                                        int((np.log10(maxP/minP))/np.log10(1 + dP) + 1))

            factor = np.sqrt(1 + dP / (1 - overlap))
            self.fbounds = [[1 / (c / factor),
                             1 / (c * factor)]
                               for c in self.centres]

        elif spacing_type == 'fixed_frequency':
            self.dP = dP if dP else 0.01
            self.centres = 1 / np.arange(1 / maxP, 1 / minP, dP)

            factor = dP / (2.0 - overlap)
            self.fbounds = [[1 / c - factor, 1 / c + factor]
                            for c in self.centres]
            
    def run(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        zero_crosses = []
        snrs = []
        v_errors = []

        for C, B in zip(self.centres, self.fbounds):
            trace = self.trace.copy()

            trace.data = self._bandpass_filter(trace.data, B[0], B[1])
            snr = self._get_signal_snr(trace, C)

            # transform into velocity space
            V = self._signal_time_to_velocity(trace.data, C)

            # mask into velocity range of interest
            valid_mask = (V >= self.minV) & (V <= self.maxV)
            trace.data = trace.data[valid_mask]
            V = V[valid_mask]

            # find zero crossing points
            crossing_indices = np.where(np.diff(np.sign(trace.data)) != 0)[0]

            # take gradient of trace for error analysis
            gradient = abs(np.gradient(trace.data, axis=1))

            CI_err = []

            for CI in crossing_indices:
                grad_ci = gradient[CI]

                left_idx = grad_ci
                right_idx = grad_ci

                while gradient[left_idx] > grad_ci * 0.8 and left_idx > 0:
                    left_idx -= 1
                while gradient[right_idx] > grad_ci * 0.8 and right_idx < len(gradient):
                    right_idx += 1
                
                CI_err.append([V[left_idx], V[right_idx]])

            # Get the velocity values at zero crossings
            crossing_velocities = V[crossing_indices]

            v_errors.append(CI_err)
            snrs.append(snr)
            zero_crosses.append(crossing_velocities)

        return None
        
    
    def _get_snr_filters(self):

        periods = []
        fbounds = []

        period = self.maxP

        while period > self.minP:
            snr = 0.0
            filt_width = 0.00

            while snr < self.snr_cutoff:
                filt_width += 0.01

                freq_max = 1.0 / (period - filt_width * period)
                freq_min = 1.0 / (period + filt_width * period)

                if freq_max > 1 / self.minP:
                    break

                trace = self.trace.copy()

                trace.data = self._bandpass_filter(trace.data, freq_min, freq_max)

                snr = self._get_signal_snr(trace, period)

                if filt_width >= 1 / self.minP and len(periods) == 0:
                    periods.clear()
                    period *= 0.8           # try starting again at 80% of the period
                    filt_width = 0.00
            
            periods.append(period)
            fbounds.append([freq_min, freq_max])


            if freq_max > 1 / self.minP:
                break

            period -= abs((1.0 / freq_max) - period) * (1- self.overlap)

        return periods, fbounds
    
    def _signal_time_to_velocity(self, trace, centre_period):

        t = trace.times()
        mask = t > 0
        v = self.dist / (t[mask]- centre_period / 8) # Yao et al 2006, phase velocity correction

        return v
    
    def _get_signal_snr(self, trace, centre_period):

        v = self._signal_time_to_velocity(trace, centre_period)
        data = trace.data

        # Noise: RMS of the trace outside the zone of interest
        signal_mask = (v >= self.minV) & (v <= self.maxV)
        noise_mask  = (v < self.minV / self.snr_pad) | (v > self.maxV * self.snr_pad)

        rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
        rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))

        return rms_signal / (rms_noise + 1e-12)
    
    def _bandpass_filter(self, data, fmin, fmax):

        nyquist = 0.5 * self.trace.stats.sampling_rate
        low = fmin / nyquist
        high = fmax / nyquist

        if self.filt_type == 'bessel':
            # Bessel bandpass filter
            filt = signal.bessel(N=6, Wn=[low, high], btype='bandpass', output='sos')
    
        elif self.filt_type == 'butterworth':
            # Butterworth bandpass filter
            filt = signal.butter(N=6, Wn=[low, high], btype='bandpass', output='sos')

        return(signal.sosfiltfilt(filt, data))
    
    def _symmetrise_trace(self, fudges = None):

        trace = self.trace_raw.copy()

        d = trace.data.astype(float)
        d /= np.max(np.abs(d))

        fudge = 0

        if self.sta1 in fudges.keys():
            fudge += fudges[self.sta1]
        if self.sta2 in fudges.keys():
            fudge -= fudges[self.sta2]

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

            