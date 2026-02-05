import os
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from asnlib.dispersion.utils import (
    symmetrise_trace,
    get_attribute,
    signal_time_to_velocity,
    get_signal_snr,
    get_wavelengths
)

from asnlib.dispersion.plot import(
    plot_points,
    plot_background
)

class ftan:

    def __init__(self,
                 sta1,
                 sta2,
                 cc_file:str,
                 **kwargs):
        """
        Initialise FTAN by reading in CC and defining stations

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

        del self.trace_raw

    def configure(self,
                  spacing_type:str = 'snr',
                  filter_type:str = 'butterworth',
                  method:str = 'phase',
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
            method (str, optional): Defaults to 'phase'.
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
        self.method = method
        self.spacing_type = spacing_type
        self.filter_type = filter_type
        self.snr_cutoff = snr_cutoff
        self.snr_pad = snr_pad
        self.wavelength_cutoff = wavelength_cutoff
        self.overlap = overlap

        if spacing_type == 'snr':
            self.dP = dP if dP else 0.01
            self.periods, self.fbounds = self._get_snr_filters()

        elif spacing_type == 'fixed_period':
            self.dP = dP if dP else 0.25
            self.periods = np.arange(self.minP, self.maxP + self.dP, self.dP)

            factor = self.dP / (2.0 - self.overlap)
            self.fbounds = [[1.0 / (c + factor),
                            1.0 / max(c - factor, 1e-6)]
                              for c in self.periods]
        
        elif spacing_type == 'logarithmic':
            self.dP = dP if dP else 0.05
            self.periods = np.logspace(np.log10(self.minP), np.log10(self.maxP),
                                        int((np.log10(self.maxP/self.minP))/np.log10(1 + self.dP) + 1))

            factor = np.sqrt(1 + self.dP / (1 - self.overlap))
            self.fbounds = [[1 / (c * factor),
                             1 / (c / factor)]
                               for c in self.periods]

        elif spacing_type == 'fixed_frequency':
            self.dP = dP if dP else 0.01
            self.periods = 1 / np.arange(1 / self.maxP, 1 / self.minP, self.dP)

            factor = self.dP / (2.0 - overlap)
            self.fbounds = [[1 / c - factor, 1 / c + factor]
                            for c in self.periods]
            
    def run(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        velocity_points = []
        snrs = []
        v_errors = []
        wavelengths = []

        vtraces = []

        for C, B in zip(self.periods, self.fbounds):
            trace = self.trace.copy()

            trace.data = self._bandpass_filter(trace.data, B[0], B[1])

            if self.method == 'group':
                trace.data = np.abs(signal.hilbert(trace.data))

            snr = get_signal_snr(self, trace, self.method, centre_period = C)

            # transform into velocity space
            V, vmask = signal_time_to_velocity(self, trace, self.method, C)
            trace.data = trace.data[vmask]

            # mask into velocity range of interest
            valid_mask = (V >= self.minV) & (V <= self.maxV)
            trace.data = trace.data[valid_mask]
            V = V[valid_mask]

            if self.method == 'phase':
                # find zero crossing points
                crossing_velocities = []
                crossing_indices = np.where(np.diff(np.sign(trace.data)) != 0)[0]

                # take gradient of trace for error analysis
                gradient = abs(np.gradient(trace.data))

                CI_err = []

                for CI in crossing_indices:
                    grad_ci = gradient[CI]

                    left_idx = CI
                    right_idx = CI

                    while gradient[left_idx] > grad_ci * 0.8 and left_idx > 0:
                        left_idx -= 1
                    while gradient[right_idx] > grad_ci * 0.8 and right_idx < len(gradient) - 1:
                        right_idx += 1
                    
                    CI_err.append([V[CI] - V[left_idx], V[right_idx] - V[CI]])
                    crossing_velocities.append(V[CI])

            elif self.method == 'group':
                # find maxima velocity
                crossing_velocities = [V[np.argmax(trace.data)]]

                CI_err = [
                    [crossing_velocities[0] - min(V[trace.data > 0.8 * max(trace.data)]),
                           max(V[trace.data > 0.8 * max(trace.data)]) - crossing_velocities[0]]
                           ]

            wavelengths.append([get_wavelengths(self.dist, C, v) 
                                for v in crossing_velocities])

            v_errors.append(CI_err)
            snrs.append(snr)
            velocity_points.append(crossing_velocities)

            vtraces.append([V,trace])

        self.velocity_points = velocity_points
        self.snrs = snrs

        self.v_errors = v_errors
        self.wavelengths = wavelengths

        self.vtraces = vtraces

    def plot(self, xscalelog:bool=False, ax=None, pick_interactive=False, plot_background=False):

        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            ax = ax

        # flatten all data
        all_centres = []
        all_zcs = []
        all_colors = []
        all_xerr_lower = []
        all_xerr_upper = []
        all_yerr = []

        for C, ZCs, Ves, Pe, Ws, SNR in zip(self.periods, self.velocity_points, 
                                        self.v_errors, self.fbounds, 
                                        self.wavelengths, self.snrs):
            # Compute x errors once for this group (same Pe for all ZCs)
            xerr = np.abs(1 / np.array(Pe) - C)
            
            for ZC, Ve, W in zip(ZCs, Ves, Ws):
                # Compute y errors for this point
                yerr = np.abs(Ve)
                
                all_centres.append(C)
                all_zcs.append(ZC)
                all_colors.append('k' if W > self.wavelength_cutoff and SNR > self.snr_cutoff else 'r')
                all_xerr_lower.append(xerr[0])
                all_xerr_upper.append(xerr[1])
                all_yerr.append(yerr)

        # Convert to arrays
        all_centres = np.array(all_centres)
        all_zcs = np.array(all_zcs)
        all_yerr = np.array(all_yerr).T  # Shape: (2, N)
        all_xerr = np.array([all_xerr_lower, all_xerr_upper])  # Shape: (2, N)

        # Separate by color
        mask_black = np.array(all_colors) == 'k'

        # Plot all black points at once
        if mask_black.any():
            ax.errorbar(all_centres[mask_black], all_zcs[mask_black], 
                        yerr=all_yerr[:, mask_black], xerr=all_xerr[:, mask_black],
                        fmt='.', color='k', linestyle='',elinewidth=0.2)

        # Plot all red points at once
        if (~mask_black).any():
            ax.errorbar(all_centres[~mask_black], all_zcs[~mask_black],
                        yerr=all_yerr[:, ~mask_black], xerr=all_xerr[:, ~mask_black],
                        fmt='.', color='r', linestyle='',elinewidth=0.2)

        if xscalelog:    
            ax.set_xscale('log')

        ax.set_xlim(self.minP, self.maxP)
        ax.set_ylim(self.minV, self.maxV)

        ax.set_title("FTAN of "+get_attribute(self.sta1,'sta')+'-'+
                        get_attribute(self.sta2,'sta')+', distance: '+
                        str(round(self.dist))+'m')
        
        if plot_background:

            try:    
                from scipy.interpolate import RegularGridInterpolator

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
            except:
                pass

        if pick_interactive:

            class InteractivePicker:
                """Handles interactive ridge picking"""
                def __init__(self, parent, ax, fig):
                    self.parent = parent
                    self.ax = ax
                    self.fig = fig
                    self.c_click = []
                    self.T_click = []
                    self.line = None
                    self.locked = False  # Store locked state here
                
                def on_click(self, event):
                    if self.locked:
                        return
                    
                    self.c_click.append(event.ydata)
                    self.T_click.append(event.xdata)

                    print(self.c_click)
                    
                    if len(self.c_click) >= 2:
                        clicks = [[T, c] for T, c in zip(self.T_click, self.c_click)]
                        try:
                            P_start = self.parent._find_closest(
                                list(self.parent.centres), 
                                min([c[0] for c in clicks])
                            )
                            P_end = self.parent._find_closest(
                                list(self.parent.centres), 
                                max([c[0] for c in clicks])
                            )
                            
                            P_start_idx = list(self.parent.centres).index(P_start)
                            P_end_idx = list(self.parent.centres).index(P_end)
                            
                            v_start = self.parent._find_closest(
                                self.parent.velocity_points[P_start_idx], 
                                clicks[np.argmin([c[0] for c in clicks])][1]
                            )
                            v_end = self.parent._find_closest(
                                self.parent.velocity_points[P_end_idx], 
                                clicks[np.argmax([c[0] for c in clicks])][1]
                            )
                            
                            self.ridge = self.parent._ridge_picker(
                                bound_points=[[P_start, v_start], [P_end, v_end]]
                            )

                            self.parent.ridge = self.ridge
                            
                            if self.line is not None:
                                self.line.remove()
                            
                            self.line, = self.ax.plot(
                                [r[0] for r in self.ridge], 
                                [r[1] for r in self.ridge],
                                color="orange", 
                                zorder=5, 
                                linewidth=5
                            )
                            self.fig.canvas.draw()
                            
                        except Exception as e:
                            print("Could not find curve - try again...")
                            print(e)
                        finally:
                            self.c_click.clear()
                            self.T_click.clear()
                
                def on_key(self, event):
                    if event.key == 'space':
                        self.locked = not self.locked
                        print("locked =", self.locked)

                    if event.key == 'enter':
                        if not hasattr(self, 'ridge') or self.ridge is None:
                            self.parent.ridge = None
                        plt.close('all')

            self._picker = InteractivePicker(self, ax, fig)
            fig.canvas.mpl_connect('button_press_event', self._picker.on_click)
            fig.canvas.mpl_connect('key_press_event', self._picker.on_key)

            plt.show() 
    
    def _get_snr_filters(self):

        periods = []
        fbounds = []

        period = self.maxP

        while period > self.minP:
            snr = 0.0
            filt_width = self.dP

            while snr < self.snr_cutoff:

                freq_max = 1.0 / (period - filt_width * period)
                freq_min = 1.0 / (period + filt_width * period)

                if freq_max > 1 / self.minP:
                    break

                trace = self.trace.copy()

                trace.data = self._bandpass_filter(trace.data, freq_min, freq_max)

                if self.method == 'group':
                    trace.data = np.abs(signal.hilbert(trace.data))

                snr = get_signal_snr(self,trace,self.method,centre_period=period)

                filt_width += 0.01

                if filt_width >= 2.5 / period and len(periods) == 0:
                    period *= 0.8           # try starting again at 80% of the period
                    filt_width = self.dP
                    snr = 0.0
                
                elif filt_width >= 5 / period and period > self.minP:
                    period *= 0.8
                    filt_width = self.dP
                    snr = 0.0
                
            
            periods.append(period)
            fbounds.append([freq_min, freq_max])


            if freq_max > 1 / self.minP:
                break

            period -= abs((1.0 / freq_max) - period) * (1- self.overlap)

        return periods, fbounds
    
    def _bandpass_filter(self, data, fmin, fmax):

        nyquist = 0.5 * self.trace.stats.sampling_rate
        low = fmin / nyquist
        high = fmax / nyquist

        if self.filter_type == 'bessel':
            # Bessel bandpass filter
            filt = signal.bessel(N=6, Wn=[low, high], btype='bandpass', output='sos')
    
        elif self.filter_type == 'butterworth':
            # Butterworth bandpass filter
            filt = signal.butter(N=6, Wn=[low, high], btype='bandpass', output='sos')

        return(signal.sosfiltfilt(filt, data))

    def _find_closest(self, data, ref):
        return data[np.argmin([np.abs(d-ref) for d in data])]

    def _ridge_picker(self, regularise=True, reg_step=0.25,
                     spline_type='lsqr', bound_points=None):

        if bound_points:
            bp0 = bound_points[0]
            bp1 = bound_points[1]

            ridge = [[bp0[0],bp0[1]]]
            point = bp0

            for P in sorted(self.periods):
                if P <= bp0[0]:
                    continue
                elif P > bp1[0]:
                    break
                    
                V = self.velocity_points[list(self.periods).index(P)]
                
                V_C = self._find_closest(V, point[1])
                point = [P, V_C]
                ridge.append(point)

        if regularise:
            r_p = [r[0] for r in ridge]
            r_v = [r[1] for r in ridge]

            if spline_type == 'lsqr':
                from scipy.interpolate import generate_knots, make_lsq_spline

                knots = list(generate_knots(r_p,r_v, s=3000))

                for t in knots[::3]:
                    spl = make_lsq_spline(r_p,r_v, t)

            # reregularise to 0.25s spacing
            output_step = reg_step
            rounded_min = np.ceil(min(r_p) / output_step) * output_step
            rounded_max = np.floor(max(r_p) / output_step) * output_step

            output_range = np.arange(rounded_min, rounded_max + output_step, output_step)

            ridge.clear()
            for step in output_range:  
                ridge.append([step, float(spl(step))])

        return(ridge)
            