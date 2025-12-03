import pandas as pd
from msnoise.api import *
import obspy
import os

##############
# PARAMETERS #
##############

msnoise_dir = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-main/"
out_csv = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-main/nov_all_pairs.csv"

##############

os.chdir(msnoise_dir)

session = connect()
station_pairs = get_station_pairs(session, used=True)

data = []
for station1, station2 in station_pairs:

    if station1.sta == station2.sta:
        continue
            
    distance, az, baz = obspy.geodetics.base.gps2dist_azimuth(station1.Y, station1.X,station2.Y, station2.X)

    data.append({
        "station1": station1.sta,
        "station2": station2.sta,
        "ZZ": "TRUE",
        "TT": "TRUE",
        "gcm": round(distance, 3),
        "az": az,
        "baz": baz,
    })

df = pd.DataFrame(data)

df.to_csv(out_csv, index=False)