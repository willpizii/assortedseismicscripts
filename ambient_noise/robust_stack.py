import numpy as np
from obspy import read, Trace, UTCDateTime
import os
import glob
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x
import pandas as pd
from asnlib.stack import robust_stack

##############
# PARAMETERS #
##############

input_dir = '/space/wp280/CCFRFR/STACKS/01/001_DAYS/TT/*'
stack_dir = '/space/wp280/CCFRFR/robust/CC/TT'
egf_dir = '/space/wp280/CCFRFR/robust/EGF/TT'

##############

paths = sorted(glob.glob(input_dir))
os.makedirs(stack_dir, exist_ok=True)
os.makedirs(egf_dir, exist_ok=True)

print("Stacking and writing output files...")

for path in tqdm(paths):

    files = sorted(glob.glob(os.path.join(path, '*.MSEED')))
    streams = [read(f)[0] for f in files]

    # Stack data into array (N traces x M samples)
    ncfs = np.stack([tr.data for tr in streams])

    stack, weight = robust_stack(ncfs)

    ref = streams[0]

    fname = path.split('/')[-1]
    stack_file = os.path.join(stack_dir, fname+ '.mseed')

    tr_out = Trace(data=stack.astype(np.float32), header=ref.stats)
    tr_out.stats.starttime = ref.stats.starttime
    tr_out.write(stack_file, format="MSEED")

    x = stack.copy()
    npts = len(x)
    delta = streams[0].stats.delta
    fs = streams[0].stats.sampling_rate
    comp = streams[0].stats.channel

    mid = (npts - 1) // 2
    x1 = np.flip(x[0:mid])
    x2 = x[mid+1:]

    # symmetric average
    xout = (x1 + x2) / 2

    # derivative
    xout = -np.diff(xout) / delta

    # write as ObsPy trace
    outStats = streams[0].stats.copy()
    outStats.npts = len(xout)
    egf_trace = Trace(data=xout.astype(np.float32), header=outStats)

    egf_file = os.path.join(egf_dir, fname+ '.mseed')

    egf_trace.write(egf_file, format="MSEED")

try:
    from cowsay import meow
except:
    def meow(x):
        print(x)

meow("All stacks and egfs complete and written")
