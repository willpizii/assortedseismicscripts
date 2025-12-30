import pandas as pd, numpy as np, matplotlib.pyplot as plt
import json
from pyFMST import fmst

picks_json = "/space/wp280/CCFRFR/ZZ_PICKS.json"
pairs_csv  = "/space/wp280/CCFRFR/nov_all_pairs.csv"
stations   = "/space/wp280/CCFRFR/frfr_stations.csv"

outfile = "zz-azi.png"

phase_vel = 2.7977474632885455

############
tomo = fmst(path="/space/wp280/FMST/FMST", templates="/home/wp280/Documents/pyFMST/templates")

tomo.load_stations(stations)
tomo.load_velocity_pairs(picks_json, phase_vel, ignore_stations=["LAMB","SMAL","THOR"])
tomo.read_station_pairs(pairs_csv, drop=True)

df = tomo.station_pairs_complete

bins = np.arange(0, 181, 10)
centers = bins[:-1] + 5

az = df["az"].values
az_fold = np.mod(az, 180.0)
df["az_fold"] = az_fold

df["az_bin"] = pd.cut(
    df["az_fold"],
    bins=bins,
    labels=centers,
    right=False,
    include_lowest=True
)

# Mean and std of velocity per bin
g = (
    df.groupby("az_bin", observed=False)["vel"]
      .agg(["mean", "std"])
      .dropna()
      .reset_index()
)

theta = np.deg2rad(g["az_bin"].astype(float))
mean_vel = g["mean"].values
std_vel = g["std"].values

# Non-zero radial baseline
baseline = mean_vel.min() * 0.9
height = mean_vel - baseline

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

ax.bar(
    theta,
    height,
    width=np.deg2rad(10),
    bottom=baseline,
    yerr=std_vel,
    align="center",
    edgecolor="k",
    alpha=0.7
)

# Duplicate to cover full circle visually (symmetry)
ax.bar(
    theta + np.pi,
    height,
    width=np.deg2rad(10),
    bottom=baseline,
    yerr=std_vel,
    align="center",
    edgecolor="k",
    alpha=0.7
)

# Polar formatting
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_rlim(baseline, mean_vel.max() + std_vel.max())
ax.set_rlabel_position(135)
ax.set_ylabel("Velocity", labelpad=20)

plt.savefig(outfile)
plt.show()
