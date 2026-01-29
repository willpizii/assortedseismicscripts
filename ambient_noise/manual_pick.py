from obspy import read
from obspy.signal.filter import envelope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
import scipy
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from tqdm import tqdm
import json

##############
# PARAMETERS #
##############

stack_dir = f"/space/wp280/CCFRFR/linear/CC/{args.comp}"
station_pairs = "/space/wp280/CCFRFR/nov_all_pairs.csv"
sta1 = args.sta1 or "SVIN"
sta2 = args.sta2 or "KEFE"
net = "RK"

json_file = None # f'/space/wp280/CCFRFR/{args.comp}_OVERLAP_REG_PICKS.json'
outfile = None # 'single_dispersion_linear.png'

method = 'phase'
f_type = 'snr'
filt_type = 'butterworth'

maxv = 4000
minv = 1000
maxP = 15
minP = 0.5
overlap = 0.75
dP = 0.05

snr_thresh = float(args.snr)

step_jump = 2
wavelengths = 1.0
output_step = 0.25

fudges = {'LAMB':4.0, 'SMAL': 2.0, 'THOR': 3.0} # or None
sample_rate = 50.0

##############

if sta1 > sta2:
    sta1, sta2 = sta2, sta1
    
seps = pd.read_csv(station_pairs)

st = read(f"{stack_dir}/{net}_{sta1}_{net}_{sta2}.mseed")

tr = st[0]
d = tr.data.astype(float)
d /= np.max(np.abs(d))

if sta1 in fudges.keys():
    fudge = fudges[sta1]
    if sta2 in fudges.keys():
        fudge -= fudges[sta2]

elif sta2 in fudges.keys():
    fudge = - fudges[sta2]

else:
    fudge = None

if not fudge:
    mid = d.size // 2
    d = d[:2*mid]
    s = 0.5 * (d[mid:] + d[:mid][::-1])
    tr.data = s
else:
    # Shift the midpoint by fudge_s (in seconds)
    shift_samples = int(fudge * sample_rate)
    mid = d.size // 2
    new_mid = mid + shift_samples

    # Ensure the new midpoint is within bounds
    if new_mid < 0 or new_mid >= d.size:
        raise ValueError("fudge shifts the midpoint out of bounds.")

    # Symmetrize around the new midpoint
    left = d[new_mid:]
    right = d[:new_mid][::-1]

    # Trim to the same length as the original symmetrized trace
    min_len = min(len(left), len(right))
    s = 0.5 * (left[:min_len] + right[:min_len])
    tr.data = s

dist = float(seps[(seps['station1'] == sta1) & (seps['station2'] == sta2)]['gcm'].iloc[0])

# Period range and other initializations
if f_type == 'fixed':
    periods = np.arange(minP, maxP + dP, dP)
elif f_type == 'relative':
    periods = np.logspace(np.log10(minP), np.log10(maxP), int((np.log10(maxP/minP))/np.log10(1 + dP) + 1))
elif f_type == 'inverse':
    freqs = np.arange(1 / maxP, 1 / minP, dP)
    periods = 1 / freqs

elif f_type == 'snr':

    fsts_snr = {}
    periods = []
    period = maxP

    while period > minP:
        snr_last = 0.0
        filt_width = 0.00

        while snr_last < snr_thresh:
            filt_width += 0.01

            freq_max = 1.0 / (period - filt_width * period)
            freq_min = 1.0 / (period + filt_width * period)

            if freq_max > 1 / minP:
                break

            if filt_type == 'bessel':
                nyquist = 0.5 * st[0].stats.sampling_rate
                low = freq_min / nyquist
                high = freq_max / nyquist

                # Design Bessel bandpass filter
                sos = signal.bessel(N=6, Wn=[low, high], btype='bandpass', output='sos')

                # Apply zero-phase filter to your stream
                fst = st.copy()
                for tr in fst:
                    # Apply zero-phase filter (forward-backward)
                    tr.data = signal.sosfiltfilt(sos, tr.data)
            elif filt_type == 'butterworth':
                fst = st.copy().filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=6, zerophase=True)
            
            tr = fst[0]

            t = tr.times()
            mask = t > 0
            v = dist / t[mask]
            data = tr.data[mask]

            # Noise: RMS of the trace outside the zone of interest
            signal_mask = (v >= minv) & (v <= maxv)
            noise_mask  = (v < minv / 2) | (v > maxv * 2)

            rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
            rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
            snr_last = rms_signal / (rms_noise + 1e-12)
        
        fsts_snr[period] = {"stream": fst, "fmin": freq_min, "fmax": freq_max, "fcentre": 1.0/period} 
        periods.append(period)

        if freq_max > 1 / minP:
            break

        period -= abs((1.0 / freq_max) - period) * (1- overlap) 
        

fsts = {}
vgrid = np.linspace(minv, maxv, 500)  # velocities in m/s

# Bandpass filter and FTAN processing
for P0 in periods:
    if f_type == 'fixed':
        half = dP / (2.0 - overlap)
        P_low = max(P0 - half, 1e-6)
        P_high = P0 + half

        freq_min = 1.0 / P_high
        freq_max = 1.0 / P_low 
    
    elif f_type == 'relative':
        factor = np.sqrt(1 + dP / (1 - overlap))

        P_low = P0 / factor
        P_high = P0 * factor

        freq_min = 1.0 / P_high
        freq_max = 1.0 / P_low
    
    elif f_type == 'inverse':
        half = dP / (2.0 - overlap)

        freq_min = 1 / P0 - half
        freq_max = 1 / P0 + half

    elif f_type == 'snr':
        fsts = fsts_snr
        break


    if filt_type == 'bessel':
        nyquist = 0.5 * st[0].stats.sampling_rate
        low = freq_min / nyquist
        high = freq_max / nyquist

        # Design Bessel bandpass filter
        sos = signal.bessel(N=6, Wn=[low, high], btype='bandpass', output='sos')

        # Apply zero-phase filter to your stream
        fst = st.copy()
        for tr in fst:
            # Apply zero-phase filter (forward-backward)
            tr.data = signal.sosfiltfilt(sos, tr.data)
    elif filt_type == 'butterworth':
        fst = st.copy().filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=6, zerophase=True)

    if method == 'group':
        for tr in fst:
            tr.data = envelope(tr.data)

    fsts[P0] = {"stream": fst, "fmin": freq_min, "fmax": freq_max, "fcentre": 1.0/P0}

disp = []
periods_grid = []
zero_crosses = []
snrs = []
errors_x = []

for P0, fst in fsts.items():
    for tr in fst["stream"]:
        t = tr.times()
        mask = t > 0
        v = dist / t[mask]
        data = tr.data[mask]

        # Noise: RMS of the trace outside the zone of interest
        signal_mask = (v >= minv) & (v <= maxv)
        noise_mask  = (v < minv / 2) | (v > maxv * 2)

        rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
        rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
        snr = rms_signal / (rms_noise + 1e-12)
        
        if len(v) < 2:
            continue

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

        snrs.append(snr)
        disp.append(data_resampled)
        periods_grid.append(P0)

    errors_x.append([fst['fmin'], fst['fmax']])

# Convert to 2D array for plotting
disp_array = np.array(disp)
periods_array = np.array(periods_grid)

# Calculate gradient (derivative) with respect to velocity
# This highlights edges and transitions in the signal
gradient = abs(np.gradient(disp_array, axis=1))  # gradient along velocity axis

# Calculate gradient magnitude (combines horizontal and vertical gradients)
grad_period = np.gradient(disp_array, axis=0)  # gradient along period axis
grad_velocity = np.gradient(disp_array, axis=1)  # gradient along velocity axis
gradient_magnitude = np.sqrt(grad_period**2 + grad_velocity**2)

# Find zero crossings and calculate velocity errors based on gradient width
zero_crosses = []
for i, data_resampled in enumerate(disp_array):
    try:
        zc_idx = np.where(np.diff(np.sign(data_resampled)) != 0)[0]
    except Exception:
        zc_idx = []
    
    # Calculate velocity errors based on gradient width
    velocity_errors = []
    if len(zc_idx) > 0:
        # Get the gradient for this period
        grad_at_period = gradient_magnitude[i, :]
        
        for idx in zc_idx:
            # Find the peak gradient near this zero crossing
            search_range = 10  # search +/- 10 indices
            start_idx = max(0, idx - search_range)
            end_idx = min(len(grad_at_period), idx + search_range)
            
            local_grad = grad_at_period[start_idx:end_idx]
            if len(local_grad) == 0:
                velocity_errors.append(50.0)  # default error
                continue
            
            # Find local maximum gradient
            local_max_idx = np.argmax(local_grad)
            global_max_idx = start_idx + local_max_idx
            max_grad = grad_at_period[global_max_idx]
            
            # Find half-maximum points (width at half max of gradient peak)
            half_max = max_grad * 0.8
            
            # Search left
            left_idx = global_max_idx
            while left_idx > 0 and grad_at_period[left_idx] > half_max:
                left_idx -= 1
            
            # Search right
            right_idx = global_max_idx
            while right_idx < len(grad_at_period) - 1 and grad_at_period[right_idx] > half_max:
                right_idx += 1
            
            # Calculate error as half the width
            v_width = vgrid[right_idx] - vgrid[left_idx]
            v_error = v_width / 2.0
            
            # Set reasonable bounds on error
            v_error = np.clip(v_error, 10.0, 500.0)
            velocity_errors.append(v_error)
    
    zero_crosses.append({
        "period": periods_array[i],
        "zero_cross_indices": np.array(zc_idx, dtype=int),
        "velocity_errors": np.array(velocity_errors),
        "period_range": errors_x[i]
    })

# Create figure
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.05)

# Main FTAN plot
ax = fig.add_subplot(gs[0, 0])

ax.set_ylabel("Velocity (m/s)")
ax.set_xlim(minP,maxP)
ax.set_ylim(minv, maxv)

critvel = {}

for p in periods:
    critvel[p] = (1 / p) * dist

ax.fill_between(critvel.keys(), critvel.values(), y2=10000, color='grey',alpha=0.6)

# Plot zero crossings with error ellipses
for zc in zero_crosses:
    P0 = zc["period"]
    idx = zc["zero_cross_indices"]
    v_errors = zc["velocity_errors"]
    period_range = zc['period_range']
    
    if idx.size:
        P_low = 1.0 / period_range[1]
        P_high = 1.0 / period_range[0]
        
        # Period errors (width in x-direction)
        period_error_low = P0 - P_low
        period_error_high = P_high - P0
        period_error_avg = (period_error_low + period_error_high) / 2.0
        
        # Plot scatter points
        ax.scatter([P0]*len(idx), vgrid[idx], color="lime", s=8, zorder=3, 
                  edgecolors='black', linewidths=0.5)
        
        # Plot error ellipses
        for i, (v_idx, v_err) in enumerate(zip(idx, v_errors)):
            # Ellipse parameters
            # width = 2 * period_error (horizontal extent)
            # height = 2 * velocity_error (vertical extent)
            ellipse = Ellipse(
                xy=(P0, vgrid[v_idx]),
                width=2 * period_error_avg,  # Total width in period
                height=2 * v_err,  # Total height in velocity
                angle=0,  # No rotation
                edgecolor='lime',
                facecolor='lime',
                linewidth=0.8,
                alpha=0.1,
                zorder=2
            )
            ax.add_patch(ellipse)

# Plot picked curve if available
if json_file:
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        curve = data[f'{net}_{sta1}_{net}_{sta2}']
        ax.plot(curve[0], [1000 * v for v in curve[1]], color="cyan", 
                linewidth=2.5, label='Picked curve', zorder=4)
        # ax.legend()
    except:
        pass

ax.set_xscale("log")

# SNR plot
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.plot(periods_array, snrs, color="b", lw=1)
ax2.set_ylabel("SNR")
ax2.set_xlabel("Filter period (s)")
ax2.set_xlim(minP,maxP)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, np.max(snrs) * 1.1)
ax2.axhline(snr_thresh, color='r', ls='--', alpha=0.5)
plt.setp(ax.get_xticklabels(), visible=False)

# ax.set_xscale("log")
# ax.set_xlim(0.5,1.5)
# ax.set_ylim(1500,2000)

# Add colorbar
# cbar = plt.colorbar(im, ax=[ax,ax2], label='Gradient Magnitude')

title = f'FTAN of {sta1}-{sta2} : Distance {dist:.0f} m'
ax.set_title(title)

if outfile:
    plt.savefig(outfile, dpi=150, bbox_inches='tight')

c_click = []
T_click = []

def find_closest(c_list,c_ref):
    return np.argmin([np.abs(c-c_ref) for c in c_list])

def ridge_picker(zc, bound_points, regularise=True):

    bp0 = bound_points[0]
    bp1 = bound_points[1]

    ridge = [[bp0[0],bp0[1]]]
    point = bp0

    for P in sorted(np.unique([i['period'] for i in zero_crosses])):
        if P <= bp0[0]:
            continue
        elif P > bp1[0]:
            break
            
        V = vgrid[next(i['zero_cross_indices'] for i in zero_crosses if i['period'] == P)]
        
        V_C = find_closest(V, point[1])
        point = [P, V[V_C]]
        ridge.append(point)

    if regularise:
        r_p = [r[0] for r in ridge]
        r_v = [r[1] for r in ridge]

        knots = list(scipy.interpolate.generate_knots(r_p,r_v, s=3000))

        for t in knots[::3]:
            spl = scipy.interpolate.make_lsq_spline(r_p,r_v, t)

        # reregularise to 0.25s spacing
        output_step = 0.25
        rounded_min = np.ceil(min(r_p) / output_step) * output_step
        rounded_max = np.floor(max(r_p) / output_step) * output_step

        output_range = np.arange(rounded_min, rounded_max + output_step, output_step)

        ridge.clear()
        for step in output_range:  
            ridge.append([step, float(spl(step))])

    return(ridge)

locked = False

def keyhandler(event):
    global locked
    if event.key != 'space':
        locked = not locked

    print("locked = ",str(locked))

def onclick(event):
    if locked:
        return
    global c_click
    global T_click

    c_click.append(event.ydata)
    T_click.append(event.xdata)
    ax.scatter(c_click,T_click,marker='x')

    print(c_click,T_click)

    if len(c_click) >= 2:
        clicks = [[T,c] for T,c in zip(T_click, c_click)]

        try:
            P_start = periods_array[find_closest(periods_array,min([c[0] for c in clicks]))]
            P_end   = periods_array[find_closest(periods_array,max([c[0] for c in clicks]))]
            
            v_start = next(vgrid[i['zero_cross_indices']] for i in zero_crosses if i['period'] == P_start)[
                find_closest(next(vgrid[i['zero_cross_indices']] for i in zero_crosses if i['period'] == P_start),
                             clicks[np.argmin([c[0] for c in clicks])][1])]
            v_end   = next(vgrid[i['zero_cross_indices']] for i in zero_crosses if i['period'] == P_end)[
                find_closest(next(vgrid[i['zero_cross_indices']] for i in zero_crosses if i['period'] == P_end),
                             clicks[np.argmax([c[0] for c in clicks])][1])]

            ridge = ridge_picker(zero_crosses, [[P_start, v_start],[P_end, v_end]])

            c_click.clear()
            T_click.clear()

        except Exception as e:
            print("Could not find curve - try again...")
            print(e)
            c_click.clear()
            T_click.clear()

        print([r[0] for r in ridge],[r[1] for r in ridge])

        if 'line' in locals():
            line.remove()

        line, = ax.plot([r[0] for r in ridge],[r[1] for r in ridge],color="orange",zorder=5,linewidth=5)
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', keyhandler)

plt.show()