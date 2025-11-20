from obspy import read
from obspy.core import UTCDateTime
import os
from collections import defaultdict
import pandas as pd
from datetime import datetime
from tqdm import tqdm

##############
# PARAMETERS #
##############

# Path to your directory with MiniSEED files
directory = "/raid2/wp280/PhD/reykjanes/nodes/data/1/reykjanes_3"

# Input and output csvs
node_loc_csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations.csv"
output_csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_timesTEMP.csv"

# Column of serial numbers to use for each node location
col_select = 'node_1'

# Start and end column names - for if you are replacing nodes
start_col = 'start_1'
end_col = 'end_1'

##############

# Dictionary to hold start and end times per station
station_times = defaultdict(lambda: {"start": None, "end": None})

# Loop over all MiniSEED files
print("Checking miniseed files...")
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".Z.miniseed"):
        filepath = os.path.join(directory, filename)
        try:
            st = read(filepath)
            for tr in st:
                net = tr.stats.network
                sta = tr.stats.station
                station_id = f"{net}.{sta}"

                start = tr.stats.starttime
                end = tr.stats.endtime

                # Update the dictionary
                if station_times[station_id]["start"] is None or start < station_times[station_id]["start"]:
                    station_times[station_id]["start"] = start
                if station_times[station_id]["end"] is None or end > station_times[station_id]["end"]:
                    station_times[station_id]["end"] = end
        except Exception as e:
            print(f"Error reading {filename}: {e}")

stations = pd.read_csv(node_loc_csv)

for i, j in station_times.items():
    st_in = i.split('.')[1]
    start = (j['start'].strftime('%Y')+':'+
          j['start'].strftime('%j')+':'+
          j['start'].strftime('%H')+':'+
          j['start'].strftime('%M')+':'+
          j['start'].strftime('%S'))
    end = (j['end'].strftime('%Y')+':'+
          j['end'].strftime('%j')+':'+
          j['end'].strftime('%H')+':'+
          j['end'].strftime('%M')+':'+
          j['end'].strftime('%S'))

    stations.loc[stations['node_1'] == st_in, start_col] = start
    stations.loc[stations['node_1'] == st_in, end_col] = end

stations.to_csv(output_csv)

try:
    from cowsay import trex
except:
    def trex(x):
        print(x)

trex("Timings added to node csv")