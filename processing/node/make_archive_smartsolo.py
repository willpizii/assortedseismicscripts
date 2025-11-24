import pandas as pd
import os
from datetime import datetime
from obspy import read
from obspy.core import UTCDateTime
from tqdm import tqdm
import shutil
import glob
import numpy as np
import traceback, sys

##############
# PARAMETERS #
##############

data_dir = "/raid2/wp280/PhD/reykjanes/nodes/data/2"
archive_dir = "/raid2/wp280/PhD/reykjanes/nodes/archive_test"

csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_times.csv" # node SN - serial lookup

new_network = "RK"	# network code for archive and metadata
col_node = ['node_1', 'node_2']	# string if one entry, list of strings if multiple
service_code = ['00', '10']	# string for one entry, list if multiple - i.e. node replacements
format = 'REYKJANES'    # so far, "REYKJANES" only, will implement "SEISUK" soon

##############

if isinstance(col_node, str):
    col_node = [col_node]

if isinstance(service_code, str):
    service_code = [service_code]

assert(len(col_node) == len(service_code))

stations = pd.read_csv(csv)

file_list = sorted(glob.glob(f"{data_dir}/*.Z*"))
nodes = np.unique([n.split('..',1)[0].split('/')[-1] for n in file_list])

nodes_dict = {c: stations[c].tolist() for c in col_node}

start_times = {}

# Build a replacement lookup dict by checking for non-midnight mseed start times

for file in file_list:
    parts = file.split(".")
    dt_str = ".".join(parts[4:10])

    t = datetime.strptime(dt_str,"%Y.%m.%d.%H.%M.%S")

    if not (t.hour==0 and t.minute==0 and t.second==0):
        node = int(file.split('..',1)[0].split('/')[-1]) - 453000000
        d = start_times.setdefault(node,{})
        d[len(d)] = t

if max(len(v) for v in start_times.values()) != len(col_node):
    raise ValueError("Error! Number of replacements greater than number of columns given")

full_file_list = sorted(glob.glob(f"{data_dir}/*.*"))

def code_for_time(node, t):
    seq = start_times[node]
    ordered = sorted(seq.items(), key=lambda x: x[1])
    idx = max(k for k,v in ordered if t >= v)
    return idx

for filename in tqdm(full_file_list):
    filepath = filename

    try:
        t = datetime.strptime(".".join(filepath.split(".")[4:10]),"%Y.%m.%d.%H.%M.%S")
        node = int(filepath.split('..',1)[0].split('/')[-1]) - 453000000

        cft = code_for_time(node, t)

        if len(start_times[node]) < len(col_node):
            exist_cols = sorted([c for c, vals in nodes_dict.items() if node in vals])
            cfn = exist_cols[cft]

        else:
            cfn = col_node[cft]

        cde = stations.loc[stations[cfn] == int(node), "code"].iloc[0]

        row_mask = stations["code"] == cde

        if not stations.loc[row_mask, col_node].isna().any().any():
            cfs = str(service_code[col_node.index(cfn)])
        else:
            valcols = stations.loc[row_mask, col_node].notna()
            columns_with_values = valcols.columns[valcols.any()].tolist()
            cfs = str(service_code[columns_with_values.index(cfn)])

        st = read(filepath)
        for tr in st:
            start = tr.stats.starttime
            
            year = start.strftime('%Y')
            jday = start.strftime('%j')
            time_start = start.strftime('%H%M%S')
            channel = tr.stats.channel

            code = stations.loc[stations[cfn] == int(tr.stats.station), "code"].iloc[0]
            
            tr.stats.network = new_network
            tr.stats.station = code
            tr.stats.location = cfs
            
            if format == 'SEISUK':
                base_dir = os.path.join(archive_dir,year,new_network,code,f"{channel}.D") # wrong format

                newname = str(str(year)+str(jday)+"_"+str(time_start)+"_"+code+"_"+filename.split('.')[-2]+"2.m")
            else:
                base_dir = os.path.join(archive_dir,year,new_network,code,f"{channel}.D")

                newname = str(str(new_network)+'.'+str(code)+'.'+str(cfs)+'.'+str(channel)+'.D.'+str(year)+'.'+str(jday))

            os.makedirs(base_dir, exist_ok=True)
            tr.write(os.path.join(base_dir,newname), format="MSEED")

    except Exception as e:
        print("Exception type:", type(e).__name__)
        print("Exception message:", e)
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)
        print(t, node, cft, cfn)
        print("cfn:", cfn)
        print("node:", node)
        print(stations[cfn].unique())
        print((stations[cfn] == node).sum())
    

