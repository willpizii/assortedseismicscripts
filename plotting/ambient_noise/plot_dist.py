import numpy as np
from obspy import read, Trace, UTCDateTime
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stack_dir", type=str)
parser.add_argument("--pairs_csv", type=str)
parser.add_argument("-r", "--refilter", type=lambda s: [float(x) for x in s.split(",")],)
parser.add_argument("-s", "--syn_source", type=str)
parser.add_argument("-m", "--symmetric", action="store_true")
parser.add_argument("-o", "--outfile", type=str)
args = parser.parse_args()

##############
# PARAMETERS #
##############      

stack_dir  = args.stack_dir  or '/space/wp280/CCFRFR/robust/CC/ZZ'
pairs_csv  = args.pairs_csv  or '/space/wp280/CCFRFR/nov_all_pairs.csv'
refilter   = args.refilter if args.refilter else None                       # None or [low, high] frequency filters
syn_source = args.syn_source or None                                        # Only plot pairs including one station
outfile = args.outfile or 'distplot.png'
sym = args.symmetric

##############

if syn_source:
    stacks = sorted(glob.glob(os.path.join(stack_dir, f'*{syn_source}*.mseed')))
else:
    stacks = sorted(glob.glob(os.path.join(stack_dir, '*.mseed')))
plt.figure(figsize=(24, 12))

pairs = pd.read_csv(pairs_csv)

for _, f in enumerate(stacks):
    st = read(f)
    tr = st[0]

    if refilter:
        tr.filter('bandpass',  freqmin=refilter[0], freqmax=refilter[1],
                    corners=4, zerophase=True)
    
    data = tr.data / np.max(np.abs(tr.data))

    if sym:
        npts = len(data)
        mid = npts // 2
        data = data[:2*mid]
        s = 0.5 * (data[mid:] + data[:mid][::-1])
        data = s

    fname = f.split('/')[-1]
    dist = pairs[(pairs['station1'] == fname.split('.')[0].split('_')[1]) & 
                  (pairs['station2'] == fname.split('.')[0].split('_')[-1])]['gcm'].iloc[0]

    dt = tr.stats.delta  # seconds per sample
    npts = len(data)
    t = (np.arange(npts) - npts//2) * dt

    plt.plot(t, data * 1e3 + dist, color='black', linewidth=0.5)

plt.xlabel("Time [s]")
plt.ylabel("Distance / m")
plt.ylim(bottom=0)

title = "All Stacks"

if syn_source:
    title += f" including {syn_source}"

if refilter:
    title += f" refiltered between {refilter[0]}-{refilter[1]}Hz"

plt.title(title)

if outfile:
    plt.savefig(outfile)

plt.show()