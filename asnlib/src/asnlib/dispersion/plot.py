import matplotlib.pyplot as plt

def plot_points(disp, px, py, ex=None, ey=None, ax):

        # flatten all data
        all_centres = []
        all_zcs = []
        all_colors = []

        if ex:
            all_xerr = []
        if ey:
            all_yerr = []

        for C, ZCs, Ws, SNR in zip(disp.periods, disp.velocity_points, 
                                        disp.wavelengths, disp.snrs):          
            for ZC, W in zip([ZCs], [Ws]):                
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

        ax.set_title("FTAN of "+get_attribute(self.sta1,'sta')+'-'+
                        get_attribute(self.sta2,'sta')+', distance: '+
                        str(round(self.dist))+'m')

        return ax