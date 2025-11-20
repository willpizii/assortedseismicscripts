import pandas as pd
import os
from datetime import datetime
from obspy import read
from obspy.core import UTCDateTime
from tqdm import tqdm
import shutil

##############
# PARAMETERS #
##############

data_dir = "/raid2/wp280/PhD/reykjanes/nodes/data/1/reykjanes_3/"
archive_dir = "/raid2/wp280/PhD/reykjanes/nodes/archive_r"

csv = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_times.csv" # node SN - serial lookup

new_network = "RK"	# network code for archive and metadata
col_node = 'node_1'	# string if one entry, list of strings if multiple
service_code = '00'	# string for one entry, list if multiple - i.e. node replacements
format = 'REYKJANES'    # so far, "REYKJANES" only, will implement "SEISUK" soon

##############

stations = pd.read_csv(csv)

for filename in tqdm(os.listdir(data_dir)):
    filepath = os.path.join(data_dir, filename)
    try:
        st = read(filepath)
        for tr in st:
            start = tr.stats.starttime
            
            year = start.strftime('%Y')
            jday = start.strftime('%j')
            time_start = start.strftime('%H%M%S')
            channel = tr.stats.channel

            code = stations.loc[stations[col_node] == tr.stats.station, "code"].iloc[0]
            
            tr.stats.network = new_network
            tr.stats.station = code
            tr.stats.location = service_code
            
            if format == 'SEISUK':
                base_dir = os.path.join(archive_dir,year,new_network,code,f"{channel}.D") # wrong format

                newname = str(str(year)+str(jday)+"_"+str(time_start)+"_"+code+"_"+filename.split('.')[-2]+"2.m")
            else:
                base_dir = os.path.join(archive_dir,year,new_network,code,f"{channel}.D")

                newname = str(str(new_network)+'.'+str(code)+'.'+str(service_code)+'.'+str(channel)+'.D.'+str(year)+'.'+str(jday))

            os.makedirs(base_dir, exist_ok=True)
            tr.write(os.path.join(base_dir,newname), format="MSEED")

    except Exception as e:
        print(e)
    

