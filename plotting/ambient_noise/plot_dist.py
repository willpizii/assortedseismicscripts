import numpy as np
from obspy import read, Trace, UTCDateTime
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

##############
# PARAMETERS #
##############

stack_dir = '/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/robust/CC/ZZ'
pairs_csv = '/raid2/wp280/PhD/reykjanes/nodes/msnoise-test/csvs/all_stations_pairs.csv'

refilter = [0.1,1.0]		# None or [low, high] frequency filters

##############

stacks = sorted(glob.glob(os.path.join(stack_dir, '*.mseed')))
plt.figure(figsize=(12, 6))

pairs = pd.read_csv(pairs_csv)

for _, f in enumerate(stacks):
    st = read(f)
    tr = st[0]

    if refilter:
        tr.filter('bandpass',  freqmin=refilter[0], freqmax=refilter[1],
                    corners=4, zerophase=True)

    data = tr.data / np.max(np.abs(tr.data))

    fname = f.split('/')[-1]
    dist = pairs[(pairs['station1'] == fname.split('.')[0].split('_')[1]) & 
                  (pairs['station2'] == fname.split('.')[0].split('_')[-1])]['gcm'].iloc[0]

    npts = tr.stats.npts
    dt = tr.stats.delta  # seconds per sample
    t = np.arange(npts) * dt - (npts * dt) / 2

    plt.plot(t, data * 1e3 + dist, color='black', linewidth=0.5)

plt.xlabel("Time [s]")
plt.ylabel("Distance / m")
plt.title("All Stacks")
plt.show()
