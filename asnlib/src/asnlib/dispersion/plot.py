import matplotlib.pyplot as plt
import numpy as np

def plot_points(disp, ax):

        # flatten all data
        all_centres = []
        all_zcs = []
        all_colors = []

        for C, ZCs, Ws, SNR in zip(disp.periods, disp.velocity_points, 
                                        disp.wavelengths, disp.snrs):

            if isinstance(Ws, float): Ws = [Ws]
            if isinstance(ZCs, float): ZCs = [ZCs]

            for ZC, W in zip(ZCs, Ws):                
                all_centres.append(C)
                all_zcs.append(ZC)
                all_colors.append('k' if W > disp.wavelength_cutoff and SNR > disp.snr_cutoff else 'r')

        # Convert to arrays
        all_centres = np.array(all_centres)
        all_zcs = np.array(all_zcs)

        # Separate by color
        mask_black = np.array(all_colors) == 'k'

        # Plot all black points
        if mask_black.any():
            ax.scatter(all_centres[mask_black], all_zcs[mask_black],
                        marker='o', color='k')

        # Plot all red points
        if (~mask_black).any():
            ax.scatter(all_centres[~mask_black], all_zcs[~mask_black],
                        marker='.', color='r')

        ax.set_xlim(disp.minP, disp.maxP)
        ax.set_ylim(disp.minV, disp.maxV)

        return ax

def plot_traces(disp, ax, cmap='seismic'):
    
    try:    
        from scipy.interpolate import interp1d

        velocity_grid = np.linspace(disp.minV, disp.maxV, 200)
        period_grid = np.linspace(min(disp.periods), max(disp.periods), 200)

        # Initialize the grid matrix
        grid_data_irregular = np.zeros((len(disp.periods), len(velocity_grid)))

        # Fill the grid by interpolating each trace in velocity direction
        for i, (V, trace) in enumerate(disp.vtraces):
            f = interp1d(V, trace.data / max(trace.data), kind='linear', bounds_error=False, fill_value=0)
            grid_data_irregular[i, :] = f(velocity_grid)

        # Now interpolate in the period direction
        # Create interpolator from irregular period points to regular grid
        period_interpolator = interp1d(
            disp.periods, 
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
                cmap=cmap, alpha=0.6, zorder=0)
    except Exception as e:
        print(e)
        pass