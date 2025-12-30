#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
import pywt

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

minv, maxv = 1000.0, 4000.0   # velocity limits
minP, maxP = 1.0, 10.0        # period limits (s)
nscales = 80                  # number of wavelet scales

snr_threshold = 3.0
jump_threshold = [150.0,-20] # m/s max jump between adjacent period steps

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

st = read(f"{stack_dir}/{net}_{sta1}_{net}_{sta2}.mseed")

tr = st[0]
d = tr.data.astype(float)
d /= np.max(np.abs(d))

mid = d.size // 2
d = d[:2*mid]
s = 0.5 * (d[mid:] + d[:mid][::-1])
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
fc = pywt.central_frequency(wavelet_name)  # Get center freq (approx 1.0 for cmor1.5-1.0)
scales = periods * fc * fs 

# 3. Run CWT with calculated scales
coef, freqs = pywt.cwt(data, scales, wavelet_name, sampling_period=1/fs)
env = np.abs(coef)

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
    row = disp_array[:, i]
    peak_val = np.max(row)
    
    # Method: Signal / RMS of 'noise'
    # We define noise as everything below 1500 m/s and above 3500 m/s 
    # (Adjust these based on your minv/maxv)
    noise_mask = (vgrid < minv) | (vgrid > maxv)
    
    if np.any(noise_mask):
        noise_floor = np.median(row[noise_mask]) + 1e-12
        final_snrs[i] = peak_val / noise_floor
    else:
        # Fallback: if range is narrow, use mean of bottom 20% of amplitudes
        noise_floor = np.mean(np.sort(row)[:int(0.2*len(row))]) + 1e-12
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

# Optional: Clean up isolated "islands" of data points
# (Single points surrounded by NaNs are usually noise)
for i in range(1, len(picked_curve)-1):
    if np.isnan(picked_curve[i-1]) and np.isnan(picked_curve[i+1]):
        picked_curve[i] = np.nan

# -------------------
# PLOT
# -------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={"height_ratios": [8, 1.5]}, sharex=True
)

# Continuous FTAN / wavelet image
im = ax1.contourf(periods, vgrid, disp_array, levels=100, cmap="coolwarm")
ax1.plot(periods, picked_curve, 'k', lw=2)
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title(f"Continuous wavelet dispersion: {sta1}-{sta2} (dist={dist:.0f} m)")
ax1.set_xlim(minP, maxP)
ax1.set_ylim(minv,maxv)

# Rough SNR estimation
ax2.plot(periods, final_snrs, 'b', lw=1)
ax2.set_ylabel("SNR")
ax2.set_xlabel("Period (s)")
# ax2.set_xscale("log")
ax2.grid(True)
ax2.hlines(snr_threshold,0,100)
ax2.set_ylim(0, np.max(final_snrs) * 1.1)

plt.tight_layout()
plt.savefig(args.outfile, dpi=200)
plt.show()
