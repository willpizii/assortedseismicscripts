from obspy import read
from obspy.signal.filter import envelope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import gaussian_filter, maximum_filter1d
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json

##############
# PARAMETERS #
##############

net = "RK"

stack_dir = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/pws/EGF/ZZ"
station_pairs = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/csvs/pairs_ftan_check.csv"

method = 'phase'	# phase or group (though group is dodgy)
maxv = 4000 		# cutoff maximum velocity
minv = 1000 		# cutoff minimum velocity

f_type = 'fixed'    # 'fixed' or 'dependent' filter width type
maxP = 10 		    # maximum wave period to be used
dP = 0.25 		    # difference in wave periods analysed - absolute dP for 'fixed', or as fraction of period for 'dependent'

snr_thresh = 1.5	# signal to noise threshold for dispersion picking
dv_thresh = [-20,+80]	# for regional curve, minimum and maximum jump dv

step_jump = 2		# maximum number of periods skipped in picking individual dispersions
which = ['a', 'c']	# see disp_man_pick.py - which picks to read in of 'a', 'c' and 'd'

vgrid_size = 500	# velocity steps of the dense grid (for paired dispersions)
reg_vgrid_size = 50	# velocity steps of the coarse grid (for regional curve addition)

peaks = 'maxima'    # where to pick peaks on FTAN - 'maxima' or 'zero_crosses'

out_json = "picked_ridges_ROB.json"

pick_stats = True   # print statistics about the picked dispersion curves

wavelengths = 1
ref_vel = 3000

##############

seps = pd.read_csv(station_pairs)

stack_dict = {}

def proc_row(idx):
    row = seps.loc[idx]

    if row['ftan_check'] not in which:
        return

    sta1 = row['station1']
    sta2 = row['station2']    

    st = read(f"{stack_dir}/{net}_{sta1}_{net}_{sta2}.mseed")

    dist = row['gcm']

    # Period range and other initializations
    periods = np.arange(dP, dP*((maxP/dP)+1), dP)
    fsts = {}
    vgrid = np.linspace(minv, maxv, vgrid_size)  # velocities in m/s

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
            zero_cross_indices = np.sort(np.array(maxima + minima, dtype=int))
        except Exception:
            zero_cross_indices = np.array([], dtype=int)
            maxima, minima = [], []
        zero_crosses.append({
            "period": periods_array[i],
            "maxima_indices": np.array(maxima, dtype=int),
            "minima_indices": np.array(minima, dtype=int),
            "zero_cross_indices": np.array(zero_cross_indices, dtype=int)
        })
    
    ridges = []

    for i, zc in enumerate(zero_crosses):
        period = zc["period"]
        if peaks == 'zero_crosses':
            v_this = vgrid[zc["zero_cross_indices"]]  # convert indices to velocity
        elif peaks == 'maxima':
            v_this = vgrid[zc["maxima_indices"]]

        if wavelengths * period >= dist / ref_vel:
            continue

        idx = np.where(periods == period)[0][0]
        if snrs[idx] <= snr_thresh:
            continue

        if not ridges:
            # initialize ridges with first period's zero-cross velocities
            for v in v_this:
                ridges.append([(period, v)])
            continue

        # attach each v_this to existing ridges from previous period
        ridges_next = [r[:] for r in ridges]
        attached = [False] * len(v_this)

        for idx_r, ridge in enumerate(ridges):
            prev_p, prev_v = ridge[-1]
            period_step = periods_array[1] - periods_array[0]

            # find periods within one or two steps ahead
            if abs(prev_p - period) <= step_jump * period_step:
                for idx_v, v in enumerate(v_this):
                    if attached[idx_v]:
                        continue
                    if dv_thresh[0] <= (v - prev_v) <= dv_thresh[1]:
                        ridges_next[idx_r].append((period, v))
                        attached[idx_v] = True
                        break  # once attached, move to next ridge

        # start new ridges for unattached velocities
        for idx_v, v in enumerate(v_this):
            if not attached[idx_v]:
                ridges_next.append([(period, v)])

        ridges = ridges_next

    stack_dict[f'{sta1}_{sta2}'] = {
        'disp_array': disp_array,
        'zero_crosses': zero_crosses,
        'periods_array': periods_array,
        'dist': dist,
        'ridges': ridges,
        'snrs': snrs
    }

print("Creating and analysing FTAN...")

for idx in tqdm(range(len(seps))):
    proc_row(idx)



# Original grids
periods = stack_dict[next(iter(stack_dict))]['periods_array']
vgrid_fine = np.linspace(minv, maxv, stack_dict[next(iter(stack_dict))]['disp_array'].shape[1])

# Coarser velocity grid
vgrid_coarse = np.linspace(minv, maxv, reg_vgrid_size)

# Initialize density grid on coarse grid
density_coarse = np.zeros((len(vgrid_coarse), len(periods)))

# Accumulate counts
for pair, data in stack_dict.items():
    zero_crosses = data['zero_crosses']
    dist = seps.loc[seps['station1'] + '_' + seps['station2'] == pair, 'gcm'].values[0]

    for i, zc in enumerate(zero_crosses):
        period = zc['period']
        if wavelengths * period >= dist / ref_vel:
            continue

        temp_density = np.zeros(len(vgrid_fine))

        if peaks == 'zero_crosses':
            bg_points = np.concatenate([zc['maxima_indices'], zc['minima_indices']])
        elif peaks == 'maxima':
            bg_points = zc['maxima_indices']

        for idx in bg_points:
            if 5 <= idx < len(vgrid_fine)-5:
                temp_density[idx] += 1

        interp_func = interp1d(vgrid_fine, temp_density, kind='linear', bounds_error=False, fill_value=0)
        density_coarse[:, i] += interp_func(vgrid_coarse)

# Apply Gaussian filter
sigma_v, sigma_p = 2, 2
density_smooth = gaussian_filter(density_coarse, sigma=[sigma_v, sigma_p])

# Find local maxima along velocity for each period
v_max_coords = []
p_max_coords = []

for i in range(density_smooth.shape[1]):  # loop over periods (columns)
    column = density_smooth[:, i]
    local_max = (column == maximum_filter1d(column, size=3))
    v_max = vgrid_coarse[local_max]
    p_max = np.full_like(v_max, periods[i])
    
    v_max_coords.extend(v_max)
    p_max_coords.extend(p_max)

# Sort maxima by period
maxima = sorted(zip(p_max_coords, v_max_coords), key=lambda x: x[0])
periods_unique = np.unique([p for p, v in maxima])

# Build ridges
ridges = []

for p in periods_unique:
    v_this = [v for pp, v in maxima if pp == p]
    if not ridges:
        # initialize ridges with first period's maxima
        for v in v_this:
            ridges.append([(p, v)])
    else:
        # try to attach each v to existing ridge from previous period
        ridges_next = [r[:] for r in ridges]  # copy
        attached = [False]*len(v_this)
        for idx_r, ridge in enumerate(ridges):
            prev_p, prev_v = ridge[-1]
            if prev_p == p - (periods[1]-periods[0]):  # previous period
                for idx_v, v in enumerate(v_this):
                    if not attached[idx_v] and dv_thresh[0] <= (v - prev_v) <= dv_thresh[1]:
                        ridges_next[idx_r].append((p, v))
                        attached[idx_v] = True
        # start new ridges for unattached maxima
        for idx_v, v in enumerate(v_this):
            if not attached[idx_v]:
                ridges_next.append([(p, v)])
        ridges = ridges_next

# Plot density
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(
    density_smooth, origin='lower',
    extent=[periods[0], periods[-1], vgrid_coarse[0], vgrid_coarse[-1]],
    cmap='viridis', interpolation='nearest', vmax=np.max(density_smooth)/4
)

# Plot original maxima
ax.scatter(p_max_coords, v_max_coords, color='red', s=4, marker='o')

# Plot ridges and label them
goodridges = {}
grn = 0

for r0, ridge in enumerate(ridges):
    if len(ridge) < 10:
        continue

    grn += 1

    p_ridge, v_ridge = zip(*ridge)
    ax.plot(p_ridge, v_ridge, color='white', linewidth=1)
    goodridges[grn] = ridge

    # label each ridge near its first coordinate
    ax.text(p_ridge[0], v_ridge[0], str(grn),
            color='white', fontsize=8, weight='bold',
            ha='left', va='bottom')

ax.set_aspect((periods[-1]-periods[0])/(vgrid_coarse[-1]-vgrid_coarse[0]))
ax.set_xlabel("Period (s)")
ax.set_ylabel("Velocity (m/s)")
fig.colorbar(im, ax=ax, label='Density')
plt.show()

manual_regional_ridge = goodridges[int(input("Number of best curve (printed above):" ))]

manual_periods, manual_velocities = zip(*manual_regional_ridge)
manual_periods = np.array(manual_periods)
manual_velocities = np.array(manual_velocities)

selected_ridges = {}

for pair, data in stack_dict.items():
    best_ridge = None
    min_diff = np.inf

    for ridge in data['ridges']:
        # Only consider ridge periods overlapping with manual ridge
        ridge_periods, ridge_velocities = zip(*ridge)
        ridge_periods = np.array(ridge_periods)
        ridge_velocities = np.array(ridge_velocities)

        # Interpolate ridge velocities onto manual periods for comparison
        common_periods = np.intersect1d(manual_periods, ridge_periods)
        if len(common_periods) < 2:
            continue

        interp_ridge = np.interp(common_periods, ridge_periods, ridge_velocities)
        interp_manual = np.interp(common_periods, manual_periods, manual_velocities)

        # Compute difference (e.g., RMS)
        diff = np.sqrt(np.mean((interp_ridge - interp_manual)**2))
        if diff < min_diff:
            min_diff = diff
            best_ridge = ridge

    if best_ridge is not None:
        selected_ridges[pair] = best_ridge

# Plot selected ridges for all station pairs
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the selected ridges
for pair, ridge in selected_ridges.items():
    periods_ridge, v_ridge = zip(*ridge)
    ax.plot(periods_ridge, v_ridge, lw=1, color="k")

ax.set_xlabel("Period (s)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Selected Dispersion Curves Across Station Pairs")
plt.show()

ridge_dict = {}
for pair, ridge in selected_ridges.items():
    sta1, sta2 = pair.split('_')
    key = f"{net}_{sta1}_{net}_{sta2}"
    periods_ridge, v_ridge = zip(*ridge)
    ridge_dict[key] = [list(periods_ridge), [v / 1000 for v in v_ridge]]

with open(out_json, "w") as f:
    json.dump(ridge_dict, f, indent=2)

if pick_stats:
    print("Picked dispersion curve statistics")
    print(f"Number of dispersion curves found / total station pairs: {len(selected_ridges)}/{len(seps)}")
    print(f"Number of periods picked per dispersion curve: {np.mean([len(k) for k in selected_ridges.keys()])} +/- {np.std([len(k) for k in selected_ridges.keys()])}")
    print(f"Average SNR at each period (of picked curves):")
    for period in periods:    
        idx = np.where(periods == period)[0][0]
        print(period, np.mean([k['snrs'][idx] for k in stack_dict.values()]), "(",np.mean([k['snrs'][idx] for k in stack_dict.values() if k['snrs'][idx] > snr_thresh]),")")
