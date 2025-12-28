import pandas as pd, numpy as np, obspy, glob, os
from math import sin, cos
from obspy import read
try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

##############
# PARAMETERS #
##############

cc_dir = "/space/wp280/CCFRFR/STACKS/01/001_DAYS"
msn_dir = "/space/wp280/CCFRFR"

##############

os.chdir(msn_dir)

from msnoise.api import *

session = connect()
station_pairs = get_station_pairs(session, used=True)

comps = ['EE', 'EN', 'NN', 'NE']

RR_dir = os.path.join(cc_dir, 'RR')
TT_dir = os.path.join(cc_dir, 'TT')

os.makedirs(RR_dir, exist_ok=True)
os.makedirs(TT_dir, exist_ok=True)

def rotate_pair(station1, station2):
    if station1.sta == station2.sta:
        return
            
    _, az, baz = obspy.geodetics.base.gps2dist_azimuth(station1.Y, station1.X,station2.Y, station2.X)

    az *= (np.pi / 180)
    baz *= (np.pi / 180)

    rot_matrix = np.array([[-cos(az) * cos(baz), cos(az) * sin(baz), -sin(az) * sin(baz), sin(az) * cos(baz)],
                  [-sin(az) * sin(baz), -sin(az) * cos(baz), -cos(az) * cos(baz), -cos(az) * sin(baz)],
                  [-cos(az) * sin(baz), -cos(az) * cos(baz), sin(az) * cos(baz), sin(az) * sin(baz)],
                  [-sin(az) * cos(baz), sin(az)  * sin(baz), cos(az) * sin(baz), -cos(az) * cos(baz)]])
    
    files = {}
    pattern = f"{station1.net}_{station1.sta}_{station2.net}_{station2.sta}"
    
    for c in comps:
        if not os.path.exists(os.path.join(cc_dir, c, pattern)):
            print(f"Missing component {c} for {pattern}, skipping...")
            return
        files[c] = sorted(glob.glob(os.path.join(cc_dir, c, pattern,"*")))
    
    for EE_path, EN_path, NN_path, NE_path in zip(
        files["EE"], files["EN"], files["NN"], files["NE"]):

        EE = read(EE_path)[0].data
        EN = read(EN_path)[0].data
        NN = read(NN_path)[0].data
        NE = read(NE_path)[0].data

        data_vector = np.vstack([EE,EN,NN,NE])

        rot_vector = rot_matrix @ data_vector

        TT = rot_vector[0]
        RR = rot_vector[1]

        fmt = read(EE_path)[0].copy()
        fmt.data = TT.astype(fmt.data.dtype)
        
        os.makedirs(os.path.join(TT_dir, pattern), exist_ok=True)
        fmt.write(os.path.join(TT_dir, pattern,os.path.basename(EE_path)), format="MSEED")
        
        fmt = read(EE_path)[0].copy()
        fmt.data = RR.astype(fmt.data.dtype)
        
        os.makedirs(os.path.join(RR_dir, pattern), exist_ok=True)
        fmt.write(os.path.join(RR_dir, pattern,os.path.basename(EE_path)), format="MSEED")

for station1, station2 in tqdm(station_pairs):
    rotate_pair(station1, station2)
