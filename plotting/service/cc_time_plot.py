#!/usr/bin/env python
# coding: utf-8

"""
Pairwise interferogram lag scatter plot (no GPS anchoring).

- Loads MSNoise daily stacks
- Cross-correlates each day against the median reference
- Produces a pairwise lag scatter plot colored by CC
"""

from glob import glob
import os, re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from obspy import read, Stream, UTCDateTime
from obspy.signal.cross_correlation import correlate

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from concurrent.futures import ProcessPoolExecutor, as_completed
import os


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

def process_pair(ref, comp, args_dict, station, t0, t1):
    out = []

    st = load_stream(args_dict["stack_dir"], comp, station, ref)
    if len(st) < 5:
        return out

    if args_dict["lowpass_hz"]:
        st.filter("lowpass", freq=args_dict["lowpass_hz"], zerophase=True)

    ref_data = []
    data = []

    for tr in st:
        data.append(tr.data)
        d = tr.stats.starttime.datetime
        if t0 <= d <= t1:
            ref_data.append(tr.data)

    if len(data) < 3:
        return out

    if ref_data:
        ref_stack = np.median(np.vstack(ref_data), axis=0)
    else:
        ref_stack = np.median(np.vstack(data), axis=0)

    for tr in st:
        xcf = correlate(ref_stack, tr.data, args_dict["shift_samples"])
        cc = float(xcf.max())
        lag = (args_dict["shift_samples"] - xcf.argmax()) / tr.stats.sampling_rate

        x = tr.data
        n = tr.stats.npts
        centre = n // 2
        dt = 1.0 / tr.stats.sampling_rate

        left = x[:centre]
        right = x[centre + 1:]

        m = min(len(left), len(right))
        left = left[-m:]
        right = right[:m]
        right_rev = right[::-1]

        ns_lr = m - 1
        xcf_lr = correlate(left, right_rev, ns_lr)
        dt_lr = (ns_lr - xcf_lr.argmax()) * dt / 2
        cc_lr = float(xcf_lr.max())

        if ref < station:
            lag *= -1.0

        out.append(dict(
            date=tr.stats.starttime.datetime,
            station=ref,
            comp=comp,
            lag=lag,
            cc=cc,
            dt_lr=dt_lr,
            cc_lr=cc_lr
        ))

    return out

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
    p.add_argument("--min-cc-lr", type=float, default=0.4)
    p.add_argument("--plot-dir", default="PLOTS")
    p.add_argument("--symmetric-bandpass", type=list, default=[0.5,0.6])
    p.add_argument("--ref-start", type=str, default="2023-12-01")
    p.add_argument("--ref-end", type=str, default="2024-03-01")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    station = args.station.upper()
    refs = [s.strip().upper() for s in args.ref_stations.split(",")]
    comps = [c.strip().upper() for c in args.components.split(",")]
    maxlag = args.max_lag

    t0 = datetime.fromisoformat(args.ref_start)  # e.g. "2024-01-01"
    t1 = datetime.fromisoformat(args.ref_end)    # e.g. "2024-06-30"

    os.makedirs(args.plot_dir, exist_ok=True)
    for plot in ['SCATTER', 'SYMMETRICAL', 'VIOLINS']:
        os.makedirs(os.path.join(args.plot_dir, plot), exist_ok=True)

    # ------------------------------------------------------------
    args_dict = dict(
        stack_dir=args.stack_dir,
        lowpass_hz=args.lowpass_hz,
        shift_samples=args.shift_samples,
    )

    tasks = [(r, c) for r in refs for c in comps]
    rows = []

    nthreads = args.threads or os.cpu_count()

    with ProcessPoolExecutor(max_workers=nthreads) as exe:
        futures = [
            exe.submit(process_pair, r, c, args_dict, station, t0, t1)
            for r, c in tasks
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Pairwise XC"):
            rows.extend(f.result())

    df = pd.DataFrame(rows)
    df = df[df.cc >= args.min_cc]
    df.sort_values("date", inplace=True)

    # -----------------------------------------------------------------
    # PLOT CCF-TIME
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

    out = os.path.join(args.plot_dir,'SCATTER', f"{station}_pairwise_scatter.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"Saved → {out}")

    # -----------------------------------------------------------------
    # PLOT VIOLINS
    # -----------------------------------------------------------------
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
    plt.savefig(os.path.join(args.plot_dir,'VIOLINS', f"{station}_pairwise_scatter_violin.png"), dpi=180)
    plt.close()

    # -----------------------------------------------------------------
    # PLOT CCF SYMMETRY
    # -----------------------------------------------------------------

    plt.figure(figsize=(10, 5), dpi=150)

    comp_markers = {c: fr"${c}$" for c in comps}
    
    dfp = df[np.abs(df.dt_lr) < maxlag].copy()
    dfp = dfp[dfp.cc_lr >= args.min_cc_lr]
    dfp["abs_dt_lr"] = np.abs(dfp.dt_lr)

    for comp, marker in comp_markers.items():
        d = dfp[dfp.comp == comp]
        if d.empty:
            continue

        sc = plt.scatter(
            d.date,
            d.dt_lr,
            c=d.cc_lr,
            cmap="viridis_r",
            s=18,
            marker=marker,
            alpha=0.8,
            linewidths=0
        )

    # median per date (across components)
    med = (
        dfp.groupby("date", sort=True)["dt_lr"]
        .median()
    )

    # plt.plot(
    #     med.index,
    #     med.values,
    #     color="gray",
    #     alpha=0.5,
    #     linewidth=1.0,
    #     zorder=0
    # )

    plt.colorbar(sc, label="Correlation coefficient")
    plt.ylabel("Absolute Timing Error")
    plt.xlabel("Date (UTC)")
    plt.title(f"{station}: Symmetry of CCF")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out = os.path.join(args.plot_dir,'SYMMETRICAL', f"{station}_symmetrical_scatter.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"Saved → {out}")    


if __name__ == "__main__":
    main()
