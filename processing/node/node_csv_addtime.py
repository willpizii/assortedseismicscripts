from obspy import read
from obspy.core import UTCDateTime
import os
from collections import defaultdict
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import glob

##############
# PARAMETERS #
##############

# Path to your directory with archived files
directory = "/raid2/wp280/PhD/reykjanes/nodes/archive_test"

network = "RK"
deployments = 2

# Input and output csvs
node_loc_csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_times.csv"
output_csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_timesTEST.csv"

# Column of serial numbers to use for each node location
col_node = ['node_1', 'node_2']

# Start and end column names - for if you are replacing nodes
start_col = ['start_1', 'start_2']
end_col = ['end_1', 'end_2']

##############

stations_csv = pd.read_csv(node_loc_csv)
stations = stations_csv['code'].tolist()
nodes_dict = {c: stations_csv[c].tolist() for c in col_node}

out_col = [x for a, b in zip(start_col, end_col) for x in (a, b)]

# Loop over all MiniSEED files
print("Checking miniseed files...")

out_dict = {}
for station in stations:
    files = sorted(glob.glob(os.path.join(directory, "*", network, station, "DPZ.D","*")))

    groups = {}
    for f in files:
        code = f.split(".")[3]
        groups.setdefault(code, []).append(f)

    out = []
    for code, lst in groups.items():
        lst.sort()
        out.append(lst[0])
        out.append(lst[-1])
    
    out_dict[station] = out

for station, files in tqdm(out_dict.items()):
    include = [1] * deployments

    if len(files) != deployments * 2:
        exist_cols = stations_csv[stations_csv['code'] == station][col_node].notna().all().to_list()

        for idx, col in enumerate(exist_cols):
            if not col:
                include[idx] = 0

    include = [i for i in include for _ in range(2)]

    file_idx = 0

    for incl, o_c in zip(include, out_col):
        if incl == 0:
            continue

        file = files[file_idx]

        st = read(file)
        loc_code = file.split(".")[3]

        file_idx += 1

        print(st[0].stats)

        t = st[0].stats.endtime if file_idx % 2 == 0 else st[0].stats.starttime

        stations_csv.loc[stations_csv['code'] == station, o_c] = str(t.strftime('%Y')+
                                                                        ':'+t.strftime('%j')+
                                                                        ':'+t.strftime('%H')+
                                                                        ':'+t.strftime('%M')+
                                                                        ':'+t.strftime('%S'))
        
# # Dictionary to hold start and end times per station
# station_times = defaultdict(lambda: {"start": None, "end": None})


# for filename in tqdm(os.listdir(directory)):
#     if filename.endswith(".Z.miniseed"):
#         filepath = os.path.join(directory, filename)
#         try:
#             st = read(filepath)
#             for tr in st:
#                 net = tr.stats.network
#                 sta = tr.stats.station
#                 station_id = f"{net}.{sta}"

#                 start = tr.stats.starttime
#                 end = tr.stats.endtime

#                 # Update the dictionary
#                 if station_times[station_id]["start"] is None or start < station_times[station_id]["start"]:
#                     station_times[station_id]["start"] = start
#                 if station_times[station_id]["end"] is None or end > station_times[station_id]["end"]:
#                     station_times[station_id]["end"] = end
#         except Exception as e:
#             print(f"Error reading {filename}: {e}")

# for i, j in station_times.items():
#     st_in = i.split('.')[1]
#     start = (j['start'].strftime('%Y')+':'+
#           j['start'].strftime('%j')+':'+
#           j['start'].strftime('%H')+':'+
#           j['start'].strftime('%M')+':'+
#           j['start'].strftime('%S'))
#     end = (j['end'].strftime('%Y')+':'+
#           j['end'].strftime('%j')+':'+
#           j['end'].strftime('%H')+':'+
#           j['end'].strftime('%M')+':'+
#           j['end'].strftime('%S'))

#     stations_csv.loc[stations_csv['node_1'] == st_in, start_col] = start
#     stations_csv.loc[stations_csv['node_1'] == st_in, end_col] = end

stations_csv.to_csv(output_csv)

try:
    from cowsay import trex
except:
    def trex(x):
        print(x)

trex("Timings added to node csv")