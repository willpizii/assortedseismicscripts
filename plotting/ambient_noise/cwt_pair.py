#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
import pywt
import scipy

# -------------------
# ARGUMENT PARSING
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("sta1", nargs="?", default="SECO")
parser.add_argument("sta2", nargs="?", default="KEFE")
parser.add_argument("--comp", default="ZZ")
parser.add_argument("--outfile", default="cwt_dispersion.png")
args = parser.parse_args()

# -------------------
# PARAMETERS
# -------------------
stack_dir = f"/space/wp280/CCFRFR/robust/CC/{args.comp}"
station_pairs = "/space/wp280/CCFRFR/nov_all_pairs.csv"
net = "RK"

minv, maxv = 500.0, 4000.0   # velocity limits
minP, maxP = 0.5, 12.0        # period limits (s)
nscales = 250                  # number of wavelet scales

snr_threshold = 10.0
jump_threshold = [250.0,-80] # m/s max jump between adjacent period steps
len_threshold = 1.5     # wavelength cutoff number

# -------------------
# LOAD TRACE AND DISTANCE
# -------------------
sta1, sta2 = args.sta1, args.sta2
if sta1 > sta2:
    sta1, sta2 = sta2, sta1

pairs = pd.read_csv(station_pairs)
dist = float(
    pairs[(pairs["station1"] == sta1) & (pairs["station2"] == sta2)]["gcm"].iloc[0]
)

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
fc = pywt.central_frequency(wavelet_name)
scales = periods * fc * fs

# 3. Run CWT with calculated scales
coef, freqs = pywt.cwt(data, scales, 'cmor1.5-1.0', sampling_period=1/fs)

env = np.abs(coef)
for i in range(len(env)):
    env[i] /= np.max(env[i])

# -------------------
# VELOCITY-PERIOD IMAGE FOR PLOTTING
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
# SNR ESTIMATION
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
# FILTERED PICKING
# -------------------

# 1. Initial raw pick from max amplitude
t_max_idx = np.argmax(env, axis=1)
t_max = t[t_max_idx]
raw_curve = np.zeros_like(t_max)
raw_curve[t_max > 0] = dist / t_max[t_max > 0]

# 2. Create the filtered curve
picked_curve = raw_curve.copy()

# Apply SNR Mask immediately
picked_curve[final_snrs < snr_threshold] = np.nan

# Apply Jump/Gradient Mask
# We look at the difference between point i and i-1
for i in range(1, len(picked_curve)):
    v_diff = picked_curve[i] - picked_curve[i-1]
    
    # If the jump is too big, or if we want to enforce non-negative slope:
    # (Optional: add 'or (picked_curve[i] < picked_curve[i-1])' for strict prograde)
    if v_diff > jump_threshold[0] or v_diff < jump_threshold[1]:
        picked_curve[i] = np.nan 

    if picked_curve[i] >= (1 / periods[i]) * dist * 1 / len_threshold or picked_curve[i] == 0.0 :
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

if in_curve.any():

    knots = list(scipy.interpolate.generate_knots(in_periods, in_curve, s=3000))

    for t in knots[::3]:
        spl = scipy.interpolate.make_lsq_spline(in_periods, in_curve, t)

    # reregularise to 0.25s spacing
    output_step = 0.25
    rounded_min = np.ceil(min(in_periods) / output_step) * output_step
    rounded_max = np.floor(max(in_periods) / output_step) * output_step

    output_range = np.arange(rounded_min, rounded_max + output_step, output_step)

    v_reg = []
    for step in output_range:  
        v_reg.append(float(spl(step)))

else:
    v_reg = []
# -------------------
# PLOT
# -------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={"height_ratios": [8, 1.5]}, sharex=True
)

# Continuous FTAN / wavelet image
im = ax1.contourf(periods, vgrid, disp_array, levels=100, cmap="viridis")
ax1.plot(periods, picked_curve, 'k', lw=2,label="raw")
ax1.plot(periods[i_frst:i_last],spl(in_periods),label="lsqr spline")
ax1.plot(output_range, v_reg, label="regularised")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title(f"Continuous wavelet dispersion: {sta1}-{sta2} (dist={dist:.0f} m)")
ax1.set_xlim(minP, maxP)
ax1.set_ylim(minv,maxv)

critvel = {p: (1/p) * dist * 1 / len_threshold for p in periods}
ax1.fill_between(critvel.keys(), critvel.values(), y2=10000,
                  color='white',alpha=0.6,label=f"{len_threshold} wavelengths")
ax1.legend()

# Rough SNR estimation
ax2.plot(periods, final_snrs, 'b', lw=1)
ax2.set_ylabel("SNR")
ax2.set_xlabel("Period (s)")
# ax2.set_xscale("log")
ax2.grid(True)
ax2.hlines(snr_threshold,0,100,'k','--',label=f"threshold snr={snr_threshold}")
ax2.set_ylim(0, np.max(final_snrs) * 1.1)
ax2.legend()

plt.tight_layout()
plt.savefig(args.outfile, dpi=200)
plt.show()
