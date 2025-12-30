from concurrent.futures import ProcessPoolExecutor, as_completed
from asnlib.workflows.rotations import rotate_ccf
import os, numpy as np, tqdm

# convert Station objects to dicts for pickling
def station_to_dict(station):
    return {"X": station.X, "Y": station.Y, "sta": station.sta, "net": station.net}

def worker(pair_dicts):
    station1_dict, station2_dict = pair_dicts
    # rotate_ccf now expects dicts instead of Station objects
    rotate_ccf(station1_dict, station2_dict, CC_dir, RR_dir, TT_dir)

##############
# PARAMETERS #
##############

CC_dir = "/space/wp280/CCFRFR/STACKS/01/001_DAYS"
msn_dir = "/space/wp280/CCFRFR"

RR_dir = os.path.join(CC_dir, "RRPROC")
TT_dir = os.path.join(CC_dir, "TTPROC")
os.makedirs(RR_dir, exist_ok=True)
os.makedirs(TT_dir, exist_ok=True)

max_processes = 100  # choose based on CPU cores

#############

os.chdir(msn_dir)

from msnoise.api import *
session = connect()
station_pairs = get_station_pairs(session, used=True)

# convert to dicts
station_pairs_dicts = [(station_to_dict(s1), station_to_dict(s2)) for s1, s2 in station_pairs]

# run multiprocessing
with ProcessPoolExecutor(max_workers=max_processes) as exe:
    futures = [exe.submit(worker, pair) for pair in station_pairs_dicts]
    for _ in tqdm.tqdm(as_completed(futures), total=len(futures)):
        pass
