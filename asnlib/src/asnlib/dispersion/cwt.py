from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
import os
import numpy as np
import pywt

from asnlib.dispersion.utils import (
    symmetrise_trace,
    get_attribute,
    signal_time_to_velocity,
    get_signal_snr,
    get_wavelengths
)

from asnlib.dispersion.plot import plot_points, plot_traces

class cwt():

    def __init__(self,
                 sta1,
                 sta2,
                 cc_file:str,
                 **kwargs):
        """
        Initialise CWT by reading in and symmetrising CC and defining stations

        Args:
            sta1 (object or dict): station object or dictionary
            sta2 (object or dict): station object or dictionary
            cc_file (str): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
            
        self.sta1 = sta1
        self.sta2 = sta2

        self.dist, _, _ = gps2dist_azimuth(get_attribute(self.sta1,'Y'),
                                           get_attribute(self.sta1,'X'),
                                           get_attribute(self.sta2,'Y'),
                                           get_attribute(self.sta2,'X'))

        if not cc_file or not os.path.exists(cc_file):
            raise ValueError("CC file required")
        
        self.trace_raw = read(cc_file)[0]
        self.trace = symmetrise_trace(self, **kwargs)

        self.fs = self.trace_raw.stats.sampling_rate

        del self.trace_raw

    def configure(self,
                  wavelet:str = 'cmor1.5-1.0',
                  method: str = 'group',
                  minP:float = 0.5,
                  maxP:float = 10.0,
                  minV:float = 1000,
                  maxV:float = 4500,
                  nscales:int = 250,
                  snr_cutoff:float = 5.0,
                  snr_pad:float = 2.0,
                  wavelength_cutoff:float = 3.0):
        """
        Configuration for CWT dispersion analysis

        Args:
            wavelet (str, optional): Wavelet type for CWT. Defaults to 'cmor1.5-1.0'.
            minP (float, optional): _description_. Defaults to 0.5.
            maxP (float, optional): _description_. Defaults to 10.0.
            minV (float, optional): _description_. Defaults to 1000.
            maxV (float, optional): _description_. Defaults to 4500.
            nscales (int, optional): _description_. Defaults to 250.
            snr_cutoff (float, optional): _description_. Defaults to 5.0.
            snr_pad (float, optional): _description_. Defaults to 2.0.
            wavelength_cutoff (float, optional): _description_. Defaults to 3.0.
        """

        self.method = method
        self.minP = minP
        self.maxP = maxP
        self.minV = minV
        self.maxV = maxV
        self.nscales = nscales
        self.snr_cutoff = snr_cutoff
        self.snr_pad = snr_pad
        self.wavelength_cutoff = wavelength_cutoff
        self.wavelet = wavelet

        self.periods = np.logspace(np.log10(minP),np.log10(maxP),self.nscales)

        fc = pywt.central_frequency(self.wavelet)
        self.scales = self.periods * fc * self.fs

    def run(self):
        coef, _ = pywt.cwt(self.trace.data.astype(float), self.scales,
                               self.wavelet, sampling_period = 1/self.fs)
                
        env = np.abs(coef)
        for i in range(len(env)):
            env[i] /= np.max(env[i])

        snrs = []
        v_maxamps = []
        wavelengths = []

        vtraces = []

        if self.method == 'group':

            for i, P in enumerate(self.periods):
                row = env[i]

                trace = self.trace.copy()

                V, vmask = signal_time_to_velocity(self, trace, 'group')

                trace.data = row
                snr = get_signal_snr(self, trace, 'group')
                snrs.append(snr)

                trace.data = row[vmask]
                valid_mask = (V >= self.minV) & (V <= self.maxV)
                trace.data = trace.data[valid_mask]
                V = V[valid_mask]

                max_value = V[np.argmax(trace.data)]
                max_value = max_value if self.minV < max_value < self.maxV else np.nan

                v_maxamps.append(max_value)

                wavelengths.append(get_wavelengths(self.dist, P, max_value))

                vtraces.append([V,trace])

        elif self.method == 'phase':
            phase = np.angle(coef)

            for i, P in enumerate(self.periods):
                row = env[i]
                # Direct pi/2 phase correction
                rowp = np.gradient(np.angle(np.exp(1j * (phase[i] + np.pi/2))))

                trace = self.trace.copy()

                V, vmask = signal_time_to_velocity(self, trace, 'phase', centre_period=P)

                trace.data = row
                snr = get_signal_snr(self, trace, 'phase', centre_period=P)
                snrs.append(snr)

                trace.data = rowp[vmask]
                valid_mask = (V >= self.minV) & (V <= self.maxV)
                trace.data = trace.data[valid_mask]
                V = V[valid_mask]

                max_values = V[np.abs(trace.data) > 1]

                v_maxamps.append(max_values)    

                wavelengths.append([get_wavelengths(self.dist, P, MV) for MV in max_values])     

                plot_trace = self.trace.copy()
                plot_trace.data = np.angle(np.exp(1j * (phase[i] + np.pi/2)))[vmask]
                plot_trace.data = plot_trace.data[valid_mask]
            
                vtraces.append([V,plot_trace])     

        self.snrs = snrs
        self.velocity_points = v_maxamps
        self.wavelengths = wavelengths

        self.vtraces = vtraces

    def plot(self, xscalelog:bool=False, ax=None, plot_background=False):

        import matplotlib.pyplot as plt
        
        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            ax = ax

        if xscalelog:    
            ax.set_xscale('log')

        ax = plot_points(self, ax)

        ax.set_title("CWT of "+get_attribute(self.sta1,'sta')+'-'+
                        get_attribute(self.sta2,'sta')+', distance: '+
                        str(round(self.dist, -1))+'m')
        
        if plot_background:
            plot_traces(self, ax, 'seismic')