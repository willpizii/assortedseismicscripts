from obspy import read
from obspy.signal.filter import envelope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

net = "RK"

##############
# PARAMETERS #
##############

stack_dir = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/pws/EGF/ZZ"
station_pairs = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/csvs/pairs_ftan_check.csv"

method = 'phase'
maxv = 4000
minv = 1500
maxP = 10
dP = 0.25

##############

seps = pd.read_csv(station_pairs)

def review_row(idx):
    row = seps.loc[idx]

    if pd.notna(row['ftan_check']):
        return

    sta1 = row['station1']
    sta2 = row['station2']    

    st = read(f"{stack_dir}/{net}_{sta1}_{net}_{sta2}.mseed")

    dist = row['gcm']

    # Period range and other initializations
    periods = np.arange(dP, dP*((maxP/dP)+1), dP)
    fsts = {}
    vgrid = np.linspace(minv, maxv, 250)  # velocities in m/s

    # Bandpass filter and FTAN processing
    for P0 in periods:
        # avoid zero or negative period
        half = dP / 2.0
        P_low = max(P0 - half, 1e-6)
        P_high = P0 + half

        freq_min = 1.0 / P_high
        freq_max = 1.0 / P_low 

        if freq_min >= freq_max:
            continue

        fst = st.copy().filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
        if method == 'group':
            for tr in fst:
                tr.data = envelope(tr.data)

        fsts[P0] = {"stream": fst, "fmin": freq_min, "fmax": freq_max, "fcentre": 1.0/P0}

    disp = []
    periods_grid = []
    zero_crosses = []
    snrs = []

    for P0, fst in fsts.items():
        for tr in fst["stream"]:
            t = tr.times()
            mask = t > 0
            v = dist / t[mask]
            data = tr.data[mask]

            # Noise: RMS of the entire trace
            rms_noise = np.sqrt(np.mean(np.abs(data)**2))

            vmask = (v >= vgrid.min()) & (v <= vgrid.max())
            v = v[vmask]
            data = data[vmask]

            # Calculate the RMS for the signal (SNR calculation)
            # Signal: RMS between 1.5 and 4 km/s (1500 to 4000 m/s)
            rms_signal = np.sqrt(np.mean(np.abs(data)**2))
            
            if len(v) < 2:
                continue  # cannot interpolate

            # interpolate onto fixed velocity grid
            f_interp = interp1d(v, data, kind='linear', bounds_error=False, fill_value=0.0)
            data_resampled = f_interp(vgrid)

            # Correct sign flip based on previous trace
            if len(disp) > 0:
                prev = disp[-1]
                mask = (np.abs(prev) > 1e-6) & (np.abs(data_resampled) > 1e-6)
                if np.sum(mask) > 0 and np.sign(np.sum(prev[mask] * data_resampled[mask])) < 0:
                    data_resampled *= -1
            
            # Normalize the data
            data_resampled /= np.max(np.abs(data_resampled))

            # SNR calculation
            snr = rms_signal / (rms_noise + 1e-12)  # avoid division by zero

            snrs.append(snr)  # Store the SNR for this trace

            disp.append(data_resampled)
            periods_grid.append(P0)

    disp_array = np.array(disp)
    zero_crosses = []
    periods_array = np.array(periods)

    # Find maxima/minima for zero-crossing analysis
    for i, data_resampled in enumerate(disp_array):
        try:
            d1 = np.diff(data_resampled)
            tp_indices = np.where(np.diff(np.sign(d1)) != 0)[0] + 1
            d2 = np.diff(data_resampled, n=2)
            maxima = [idx for idx in tp_indices if idx-1 < len(d2) and d2[idx-1] < 0]
            minima = [idx for idx in tp_indices if idx-1 < len(d2) and d2[idx-1] > 0]
        except Exception:
            maxima, minima = [], []
        zero_crosses.append({
            "period": periods_array[i],
            "maxima_indices": np.array(maxima, dtype=int),
            "minima_indices": np.array(minima, dtype=int)
        })

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.05)

    # Main FTAN plot
    ax = fig.add_subplot(gs[0, 0])
    c = ax.contourf(periods_array, vgrid, disp_array.T, levels=100, cmap="coolwarm")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlim(0.25, 10)
    ax.axvline(1 / (3000 / dist), ls="--", color="k")

    for zc in zero_crosses:
        P0 = zc["period"]
        try:
            if len(zc["maxima_indices"]) > 0:
                ax.scatter([P0]*len(zc["maxima_indices"]),
                           vgrid[zc["maxima_indices"]], color="k", marker="o", s=4)
            if len(zc["minima_indices"]) > 0:
                ax.scatter([P0]*len(zc["minima_indices"]),
                           vgrid[zc["minima_indices"]], color="k", marker="o", s=4)
        except Exception:
            continue

    # SNR plot
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(periods_array, snrs, color="b", lw=1)
    ax2.set_ylabel("SNR")
    ax2.set_xlabel("Filter period (s)")
    ax2.set_xlim(0.25, 10)
    ax2.grid(True)
    ax2.set_ylim(0, np.max(snrs) * 1.1)
    plt.setp(ax.get_xticklabels(), visible=False)

    # carve out space for a single vertical colorbar that spans both axes:
    fig.canvas.draw()  # ensure positions are computed
    pos0 = ax.get_position()
    pos1 = ax2.get_position()

    # colorbar width and pad (figure fraction)
    cb_width = 0.03
    pad = 0.01

    # new width for both axes (steal space for colorbar)
    new_width = pos0.width - (cb_width + pad)
    new_x0 = pos0.x0

    # re-set positions so both axes share identical left/right extents
    ax.set_position([new_x0, pos0.y0, new_width, pos0.height])
    ax2.set_position([new_x0, pos1.y0, new_width, pos1.height])

    # create a colorbar axis that spans from bottom of ax2 to top of ax
    cb_bottom = pos1.y0
    cb_top = pos0.y0 + pos0.height
    cb_height = cb_top - cb_bottom
    cax = fig.add_axes([new_x0 + new_width + pad, cb_bottom, cb_width, cb_height])

    # draw colorbar into that axis
    cb = fig.colorbar(c, cax=cax, orientation="vertical")
    cb.set_label("Normalized Amplitude")

    ax.set_title(f'FTAN of {sta1}-{sta2} : Distance {dist:.0f}')

    def on_key(event):
        if event.key in ["a", "d", "c"]:
            seps.loc[idx, "ftan_check"] = event.key
            seps.to_csv(station_pairs, index=False)
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


for idx in tqdm(range(len(seps))):
    review_row(idx)

try:
    from cowsay import kitty
except:
    def kitty(x):
        print(x)

kitty("All dispersion curves have been checked")