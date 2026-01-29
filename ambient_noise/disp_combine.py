from obspy import read
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, maximum_filter1d, label
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import json, os
import pywt
import scipy

##############
# PARAMETERS #
##############

net = "RK"

stack_dir = "/space/wp280/CCFRFR/robust/CC/ZZ"
station_pairs = "/space/wp280/CCFRFR/nov_all_pairs.csv"

method = 'group'	# phase or group (phase FTAN, group CWT)
maxv = 4000 		# cutoff maximum velocity
minv = 1000 		# cutoff minimum velocity

f_type = 'snr'      # 'fixed', 'relative' or 'inverse' filter width type
maxP = 12.0	    # maximum wave period to be used
minP = 0.5          # minimum wave period to be used
dP = 0.008		    # difference in wave periods analysed - constant dP for 'fixed'; minimum dP for 'variable'

overlap = 0.80      # overlap degree between period filters - between 0.0 and 1.0

snr_thresh = 10.0	# signal to noise threshold for dispersion picking
dv_thresh = [-80,+250]	# for curves, minimum and maximum jump dv

step_jump = 2		# maximum number of periods skipped in picking individual dispersions
which = None	    # see disp_man_pick.py - which picks to read in of 'a', 'c' and 'd'. Or None to skip

vgrid_size = 500	# velocity steps of the dense grid (for paired dispersions)
reg_vgrid_size = 50	# velocity steps of the coarse grid (for regional curve addition)

nscales = 250        # scales for cwt

out_json = "/space/wp280/CCFRFR/ZZ_GROUP_NEW_PICKS.json"

reg_output = True  # regularise output steps
output_step = 0.25  # steps between output

pick_stats = True   # print statistics about the picked dispersion curves

wavelengths = 1.5
ref_vel = 3000

fudges = {'LAMB':4.0, 'SMAL': 2.0, 'THOR': 3.0} # or None
sample_rate = 50.0

filt_type = 'butterworth'

##############

seps = pd.read_csv(station_pairs)

stack_dict = {}

def proc_row(idx):
    row = seps.loc[idx]

    if which:
        if row['ftan_check'] not in which:
            return

    sta1 = row['station1']
    sta2 = row['station2']    

    if not os.path.exists(f"{stack_dir}/{net}_{sta1}_{net}_{sta2}.mseed"):
        print("No stack found for",sta1,sta2,", skipping...")
        return

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

    dist = row['gcm']

    if method == 'phase':

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
                    noise_mask  = (v < minv) | (v > maxv)

                    rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
                    rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
                    snr_last = rms_signal / (rms_noise + 1e-12)
                
                fsts_snr[period] = {"stream": fst, "fmin": freq_min, "fmax": freq_max, "fcentre": 1.0/period, "snr": snr_last} 
                periods.append(period)

                if freq_max > 1 / minP:
                    break

                period -= abs((1.0 / freq_max) - period) * (1- overlap) 
                
        fsts = {}
        vgrid = np.linspace(minv, maxv, vgrid_size)  # velocities in m/s

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


            fst = st.copy().filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=6, zerophase=True)

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

                if f_type == 'snr':
                    snrs.append(fst['snr'])
                else:
                    # Noise: RMS of the entire trace
                    signal_mask = (v >= minv) & (v <= maxv)
                    noise_mask  = (v < minv) | (v > maxv)

                    rms_signal = np.sqrt(np.mean(data[signal_mask]**2))
                    rms_noise  = np.sqrt(np.mean(data[noise_mask]**2))
                    snr = rms_signal / (rms_noise + 1e-12)

                    snrs.append(snr)  # Store the SNR for this trace
                
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

                disp.append(data_resampled)
                periods_grid.append(P0)

        disp_array = np.array(disp)
        zero_crosses = []
        periods_array = np.array(periods_grid)

        # Find maxima/minima for zero-crossing analysis
        for i, data_resampled in enumerate(disp_array):
            try:
                zc_idx = np.where(np.diff(np.sign(data_resampled)) != 0)[0]
            except Exception:
                zc_idx = []
            zero_crosses.append({
                "period": periods_array[i],
                "zero_cross_indices": np.array(zc_idx, dtype=int)
            })
        
        ridges = []
        last_periods = []

        for i, zc in enumerate(zero_crosses):
            period = zc["period"]
            v_this = vgrid[zc["zero_cross_indices"]]  # convert indices to velocity

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

            if not last_periods:
                last_periods.append(period)
                continue

            ridges_next = [r[:] for r in ridges]
            attached = [False] * len(v_this)

            # safe lookup
            ref_period = last_periods[-min(len(last_periods), step_jump)]

            # build curves at zero-gradient points on FTAN images
            for idx_r, ridge in enumerate(ridges):
                prev_p, prev_v = ridge[-1]
                
                # Find all candidate velocities within threshold
                candidates = []
                for idx_v, v in enumerate(v_this):
                    if attached[idx_v]:
                        continue
                    if prev_p >= ref_period and dv_thresh[0] <= (v - prev_v) <= dv_thresh[1]:
                        candidates.append((idx_v, v, abs(v - prev_v)))
                
                # Pick the closest velocity if multiple candidates exist
                if candidates:
                    # Sort by distance and pick the closest
                    candidates.sort(key=lambda x: x[2])
                    idx_v, v, _ = candidates[0]
                    ridges_next[idx_r].append((period, v))
                    attached[idx_v] = True

            # # build curves at zero-gradient points on FTAN images
            # for idx_r, ridge in enumerate(ridges):
            #     prev_p, prev_v = ridge[-1]
            #     v_to_eval = []

            #     if prev_p >= ref_period:
            #         for idx_v, v in enumerate(v_this):
            #             if attached[idx_v]:
            #                 continue
            #             if dv_thresh[0] <= (v - prev_v) <= dv_thresh[1]:
            #                 v_to_eval.append((period,v))
            #                 ridges_next[idx_r].append((period, v))
            #                 attached[idx_v] = True
            #                 break  # once attached, move to next ridge

            # start new ridges for unattached velocities
            for idx_v, v in enumerate(v_this):
                if not attached[idx_v]:
                    ridges_next.append([(period, v)])

            ridges = ridges_next
            last_periods.append(period)

        min_ridge_length = 5  # at least 5 points
        ridges = [r for r in ridges if len(r) >= min_ridge_length]

        if f_type == 'snr':     # force onto a regular sampling space
            for i, ridge in enumerate(ridges):
                periods_ridge, v_ridge = zip(*ridge)
                periods_ridge = periods_ridge[::-1]
                v_ridge = v_ridge[::-1]
                
                # print(ridge,periods_ridge,v_ridge)

                rounded_min = round(min(periods_ridge) / output_step) * output_step
                rounded_max = round(max(periods_ridge) / output_step) * output_step
                # print(rounded_min, rounded_max)

                output_range = np.arange(rounded_min, rounded_max + output_step, output_step)
                # print(output_range)
                v_reg = []
                idx = 0

                for step in output_range:
                    # print(step,periods_ridge)
                    idx = np.searchsorted(periods_ridge, step)
                    # print(idx)

                    if idx == 0:
                        v_reg.append(v_ridge[0])
                        continue
                    elif idx == len(periods_ridge):
                        v_reg.append(v_ridge[-1])
                        continue

                    high = periods_ridge[idx]
                    low  = periods_ridge[idx-1]
                    
                    pos = (step-low) / (high-low)
                                    
                    pos_vel = pos * v_ridge[idx] + (1-pos) * v_ridge[idx-1]

                    # print(step,high,low,pos,pos_vel)
                    
                    v_reg.append(pos_vel)
                    idx +=1

                    # print(v_reg)
                
                ridges[i] = [(p, v) for p, v in zip(output_range, v_reg)]
            
            periods_array = np.arange(minP, maxP + dP, dP)

        stack_dict[f'{sta1}_{sta2}'] = {
            'disp_array': disp_array,
            'zero_crosses': zero_crosses,
            'periods_array': periods_array,
            'dist': dist,
            'ridges': ridges,
            'snrs': snrs
        }

    elif method == 'group':

        data = tr.data.astype(float)
        data /= np.max(np.abs(data))
        fs = tr.stats.sampling_rate
        nt = len(data)
        t = np.arange(nt) / fs

        # -------------------
        # CONTINUOUS WAVELET TRANSFORM (PyWavelets)
        # -------------------
        # 1. Define desired periods
        periods = np.logspace(np.log10(minP), np.log10(maxP), nscales)

        # 2. Convert Periods to Scales
        # scale = period * center_freq * sampling_rate
        wavelet_name = 'cmor1.5-1.0'
        fc = pywt.central_frequency(wavelet_name)  # Get center freq (approx 1.0 for cmor1.5-1.0)
        scales = periods * fc * fs 

        # 3. Run CWT with calculated scales
        coef, freqs = pywt.cwt(data, scales, wavelet_name, sampling_period=1/fs)
        env = np.abs(coef)
        for i in range(len(env)):
            env[i] /= np.max(env[i])

        # -------------------
        # OPTIONAL: VELOCITY-PERIOD IMAGE FOR PLOTTING
        # -------------------
        vgrid = np.linspace(minv, maxv, 250)
        disp_array = np.zeros((len(vgrid), len(periods)))

        # Handle t=0 singularity for velocity mapping
        # Avoid index 0 of t (t=0) which results in infinite velocity
        safe_t = t.copy()
        safe_t[0] = 1e-9  # small epsilon to avoid div/0

        v_trace = dist / safe_t 

        for i, amp in enumerate(env):
            # Interpolate amplitude from Time domain to Velocity domain
            # v_trace is decreasing (Inf -> 0), so we reverse it to be increasing for np.interp
            f_interp = np.interp(vgrid, v_trace[::-1], amp[::-1], left=0.0, right=0.0)
            disp_array[:, i] = f_interp

        # -------------------
        # BETTER SNR ESTIMATION
        # -------------------
        # We calculate SNR for each period (scale)
        final_snrs = np.zeros(len(periods))

        for i in range(len(periods)):
            row = env[i]
            peak_val = np.max(row)
            
            # Use last 50% of trace as background
            noise_floor = np.mean(row[int(0.5*len(row)):]) + 1e-12
            final_snrs[i] = peak_val / noise_floor

        # -------------------
        # FILTERED PICKING (Longest Segment Only)
        # -------------------

        # 1. Initial raw pick from max amplitude
        t_max_idx = np.argmax(env, axis=1)
        t_max = t[t_max_idx]
        raw_curve = np.zeros_like(t_max)
        raw_curve[t_max > 0] = dist / t_max[t_max > 0]

        # 2. Create the filtered curve
        picked_curve = raw_curve.copy()

        # Apply SNR Mask immediately
        picked_curve[final_snrs < snr_thresh] = np.nan

        # Apply Jump/Gradient Mask
        # We look at the difference between point i and i-1
        for i in range(1, len(picked_curve)):
            v_diff = picked_curve[i] - picked_curve[i-1]
            
            # If the jump is too big, or if we want to enforce non-negative slope:
            # (Optional: add 'or (picked_curve[i] < picked_curve[i-1])' for strict prograde)
            if v_diff > dv_thresh[1] or v_diff < dv_thresh[0]:
                picked_curve[i] = np.nan 

            if picked_curve[i] >= (1 / periods[i]) * dist * 1 / wavelengths:
                picked_curve[i] = np.nan

        # Clean up isolated data points
        for i in range(1, len(picked_curve)-1):
            if np.isnan(picked_curve[i-1]) and np.isnan(picked_curve[i+1]):
                picked_curve[i] = np.nan

        # -------------------
        # B-SPLINING
        # -------------------

        # Find longest continuous segment
        length = 0
        i_trac = 1e10
        i_last = 0
        i_frst = 0
        ln_max = 0
        for i in range(0,len(picked_curve)):
            if not np.isnan(picked_curve[i]):
                length +=1
                if i < i_trac:
                    i_trac = i
            else:
                if length > ln_max:
                    i_frst = i_trac
                    i_last = i-1
                    ln_max = length
                length = 0
                i_trac=1e10

        in_periods = periods[i_frst:i_last]
        in_curve   = picked_curve[i_frst:i_last]
        in_snrs    = final_snrs[i_frst:i_last]

        if in_curve.any():
            knots = list(scipy.interpolate.generate_knots(in_periods, in_curve, s=3000))
            knotsnr = list(scipy.interpolate.generate_knots(in_periods, in_snrs, s=3000))

            for t in knots[::3]:
                spl = scipy.interpolate.make_lsq_spline(in_periods, in_curve, t)

            for t in knotsnr[::3]:
                snrpl = scipy.interpolate.make_lsq_spline(in_periods, in_snrs, t)

            # reregularise to 0.25s spacing
            rounded_min = np.ceil(min(in_periods) / output_step) * output_step
            rounded_max = np.floor(max(in_periods) / output_step) * output_step

            output_range = np.arange(rounded_min, rounded_max + output_step, output_step)

            v_reg = []
            snr_reg = []
            for step in output_range:  
                v_reg.append(float(spl(step)))
                snr_reg.append(float(snrpl(step)))
        else:
            output_range = []
            v_reg = []
            snr_reg = []

        # 4. Return in the requested format
        stack_dict[f'{sta1}_{sta2}'] = {
            'periods': output_range,
            'velocity': v_reg,
            'snrs': snr_reg
        }

print("Analysing...")

for idx in tqdm(range(len(seps))):
    proc_row(idx)

if method == 'phase':

    # Define a COMMON period grid for all station pairs
    if f_type == 'snr':
        # Create a regular grid spanning the range you want
        periods = np.arange(minP, maxP + output_step, output_step)
    else:
        # For other methods, they already share the same grid
        periods = stack_dict[next(iter(stack_dict))]['periods_array']
    
    vgrid_fine = np.linspace(minv, maxv, vgrid_size)

    # Coarser velocity grid
    vgrid_coarse = np.linspace(minv, maxv, reg_vgrid_size)

    # Initialize density grid on coarse grid
    density_coarse = np.zeros((len(vgrid_coarse), len(periods)))

    # Accumulate counts
    for pair, data in stack_dict.items():
        zero_crosses = data['zero_crosses']
        dist = seps.loc[seps['station1'] + '_' + seps['station2'] == pair, 'gcm'].values[0]

        for zc in zero_crosses:
            period = zc['period']
            
            # Find closest period in the common grid
            period_idx = np.argmin(np.abs(periods - period))
            
            # Only include if the match is close enough (within output_step)
            if np.abs(periods[period_idx] - period) > output_step:
                continue
            
            if wavelengths * period >= dist / ref_vel:
                continue

            temp_density = np.zeros(len(vgrid_fine))

            bg_points = zc['zero_cross_indices']

            for idx in bg_points:
                if 5 <= idx < len(vgrid_fine)-5:
                    temp_density[idx] += 1

            interp_func = interp1d(vgrid_fine, temp_density, kind='linear', bounds_error=False, fill_value=0)
            density_coarse[:, period_idx] += interp_func(vgrid_coarse)

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
                if prev_p == last_p:  # previous period
                    for idx_v, v in enumerate(v_this):
                        if not attached[idx_v] and dv_thresh[0] <= (v - prev_v) <= dv_thresh[1]:
                            ridges_next[idx_r].append((p, v))
                            attached[idx_v] = True
            # start new ridges for unattached maxima
            for idx_v, v in enumerate(v_this):
                if not attached[idx_v]:
                    ridges_next.append([(p, v)])
            ridges = ridges_next

        last_p = p

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

        if reg_output:
            rounded_min = round(min(periods_ridge) / output_step) * output_step
            rounded_max = round(max(periods_ridge) / output_step) * output_step

            output_range = np.arange(rounded_min,rounded_max + output_step,output_step)

            v_reg = []

            for step in output_range:
                idx = np.searchsorted(periods_ridge, step)

                if idx == 0:
                    v_reg.append(v_ridge[0])  # Use the first velocity
                    continue
                elif idx == len(periods_ridge):
                    v_reg.append(v_ridge[-1])  # Use the last velocity
                    continue

                high = periods_ridge[idx]
                low  = periods_ridge[idx-1]
                
                pos = (step-low) / (high-low)
                                
                pos_vel = pos * v_ridge[idx] + (1-pos) * v_ridge[idx-1]
                
                v_reg.append(pos_vel)
            
            ridge_dict[key] = [list(output_range), [v/1000 for v in v_reg]]
            
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

elif method == 'group':
    ridge_dict = {}

    for pair, data in stack_dict.items():
        sta1, sta2 = pair.split('_')
        key = f"{net}_{sta1}_{net}_{sta2}"
        periods_ridge = data['periods']
        v_ridge = data['velocity']
        snr_ridge = data['snrs']
        ridge_dict[key] = [list(periods_ridge), [v / 1000 for v in v_ridge]]

    with open(out_json, "w") as f:
        json.dump(ridge_dict, f, indent=2)
