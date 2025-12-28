#!/usr/bin/env python
# coding: utf-8

"""
Pairwise interferogram lag scatter plot (no GPS anchoring).

- Loads MSNoise daily stacks
- Cross-correlates each day against the median reference
- Produces a pairwise lag scatter plot colored by CC
"""

from glob import glob
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates

from obspy import read, Stream, UTCDateTime
from obspy.signal.cross_correlation import correlate

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------
def iter_pair_files(stack_dir, comp, sta_a, sta_b):
    if sta_b > sta_a:
        pat = f"{stack_dir}/001_DAYS/{comp}/*{sta_a}*{sta_b}*/*"
    else:
        pat = f"{stack_dir}/001_DAYS/{comp}/*{sta_b}*{sta_a}*/*"
    return sorted(glob(pat))


def load_stream(stack_dir, comp, sta, ref):
    st = Stream()
    for f in iter_pair_files(stack_dir, comp, sta, ref):
        try:
            tr = read(f)[0]
            date_str = os.path.basename(f).split(".")[0]
            tr.stats.starttime = UTCDateTime(date_str)
            st += tr
        except Exception:
            pass
    return st


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station", required=True)
    p.add_argument("--ref-stations", required=True)
    p.add_argument("--stack-dir", default="./STACKS/01")
    p.add_argument("--components", default="ZZ")
    p.add_argument("--max-lag", type=float, default=2.0)
    p.add_argument("--lowpass-hz", type=float, default=5.0)
    p.add_argument("--shift-samples", type=int, default=2000)
    p.add_argument("--min-cc", type=float, default=0.6)
    p.add_argument("--plot-dir", default="PLOTS")
    args = p.parse_args()

    station = args.station.upper()
    refs = [s.strip().upper() for s in args.ref_stations.split(",")]
    comps = [c.strip().upper() for c in args.components.split(",")]
    maxlag = args.max_lag

    os.makedirs(args.plot_dir, exist_ok=True)

    rows = []

    tasks = [(r, c) for r in refs for c in comps]
    it = tqdm(tasks, desc="Pairwise XC") if HAS_TQDM else tasks

    for ref, comp in it:
        st = load_stream(args.stack_dir, comp, station, ref)
        if len(st) < 5:
            continue

        if args.lowpass_hz:
            st.filter("lowpass", freq=args.lowpass_hz, zerophase=True)

        # build reference as median stack
        data = []
        for tr in st:
            data.append(tr.data)

        if len(data) < 3:
            continue

        ref_stack = np.median(np.vstack(data), axis=0)

        for tr in st:

            xcf = correlate(ref_stack, tr.data, args.shift_samples)
            cc = float(xcf.max())
            lag = (args.shift_samples - xcf.argmax()) / tr.stats.sampling_rate

            if ref < station:
                lag *= -1.0

            rows.append(dict(
                date=tr.stats.starttime.datetime,
                station=ref,
                comp=comp,
                lag=lag,
                cc=cc
            ))

    if not rows:
        print("No data found.")
        return

    df = pd.DataFrame(rows)
    df = df[df.cc >= args.min_cc]
    df.sort_values("date", inplace=True)

    # -----------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=150)

    comp_markers = {c: fr"${c}$" for c in comps}

    for comp, marker in comp_markers.items():
        d = df[(df.comp == comp) & (np.abs(df.lag) < maxlag)]
        if d.empty:
            continue

        sc = plt.scatter(
            d.date,
            d.lag,
            c=d.cc,
            cmap="viridis_r",
            s=18,
            marker=marker,
            alpha=0.8,
            linewidths=0
        )

    plt.colorbar(sc, label="Correlation coefficient")
    plt.ylabel("Lag (s)")
    plt.xlabel("Date (UTC)")
    plt.title(f"{station}: Pairwise interferogram lag scatter")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out = os.path.join(args.plot_dir, f"{station}_pairwise_scatter.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"Saved → {out}")

    # ensure day resolution
    df["day"] = pd.to_datetime(df.date).dt.normalize()

    comps_present = [c for c in comp_markers if c in df.comp.unique()]
    ncomp = len(comps_present)

    fig, axes = plt.subplots(
        ncomp, 1,
        figsize=(10, 2.2 * ncomp),
        dpi=150,
        sharex=True,
        sharey=True
    )

    if ncomp == 1:
        axes = [axes]

    for ax, comp in zip(axes, comps_present):
        d = df[(df.comp == comp) & (np.abs(df.lag) < maxlag)]
        if d.empty:
            continue

        violins = []
        positions = []

        for day, g in d.groupby("day"):
            vals = g.lag.values
            if len(vals) < 2:
                continue

            violins.append(vals)
            positions.append(mdates.date2num(day))

        if not violins:
            continue

        vp = ax.violinplot(
            violins,
            positions=positions,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for body in vp["bodies"]:
            body.set_alpha(0.6)

        ax.scatter(
            d.date,
            d.lag,
            s=8,
            alpha=0.4
        )

        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_ylabel("Lag (s)")
        ax.set_title(comp)

        ax.xaxis_date()

    axes[-1].set_xlabel("Date (UTC)")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f"{station}_pairwise_scatter_violin.png"), dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
