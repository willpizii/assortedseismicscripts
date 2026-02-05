from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
import os
import numpy as np
import pywt
from scipy.signal import argrelextrema

from asnlib.dispersion.utils import (
    symmetrise_trace,
    get_attribute,
    signal_time_to_velocity,
    get_signal_snr,
    get_wavelengths
)

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

        # flatten all data
        all_centres = []
        all_zcs = []
        all_colors = []

        for C, ZCs, Ws, SNR in zip(self.periods, self.velocity_points, 
                                        self.wavelengths, self.snrs):          
            for ZC, W in zip([ZCs], [Ws]):                
                all_centres.append(C)
                all_zcs.append(ZC)
                all_colors.append('k' if W > self.wavelength_cutoff and SNR > self.snr_cutoff else 'r')

        # Convert to arrays
        all_centres = np.array(all_centres)
        all_zcs = np.array(all_zcs)

        # Separate by color
        mask_black = np.array(all_colors) == 'k'

        # Plot all black points at once
        if mask_black.any():
            ax.scatter(all_centres[mask_black], all_zcs[mask_black],
                        marker='.', color='k')

        # Plot all red points at once
        if (~mask_black).any():
            ax.scatter(all_centres[~mask_black], all_zcs[~mask_black],
                        marker='.', color='r')

        if xscalelog:    
            ax.set_xscale('log')

        ax.set_xlim(self.minP, self.maxP)
        ax.set_ylim(self.minV, self.maxV)

        ax.set_title("FTAN of "+get_attribute(self.sta1,'sta')+'-'+
                        get_attribute(self.sta2,'sta')+', distance: '+
                        str(round(self.dist))+'m')
        
        if plot_background:

            try:    
                from scipy.interpolate import interp1d

                velocity_grid = np.linspace(self.minV, self.maxV, 200)
                period_grid = np.linspace(min(self.periods), max(self.periods), 200)

                # Initialize the grid matrix
                grid_data_irregular = np.zeros((len(self.periods), len(velocity_grid)))

                # Fill the grid by interpolating each trace in velocity direction
                for i, (V, trace) in enumerate(self.vtraces):
                    f = interp1d(V, trace.data / max(trace.data), kind='linear', bounds_error=False, fill_value=0)
                    grid_data_irregular[i, :] = f(velocity_grid)

                # Now interpolate in the period direction
                # Create interpolator from irregular period points to regular grid
                period_interpolator = interp1d(
                    self.periods, 
                    grid_data_irregular, 
                    axis=0,  # interpolate along period axis
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=0
                )

                # Interpolate onto regular period grid
                grid_data = period_interpolator(period_grid)

                # Now plot the fully regular grid
                ax.imshow(grid_data.T, aspect='auto', origin='lower',
                        extent=[period_grid.min(), period_grid.max(),
                                velocity_grid.min(), velocity_grid.max()],
                        cmap='seismic', alpha=0.6, zorder=0)
            except Exception as e:
                print(e)
                pass

        plt.show()