from obspy import read, UTCDateTime, Stream, read_inventory
import obspy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
from datetime import timedelta

# --- CONFIGURATION ---
csv_dir = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_times.csv"
data_dir = "/raid2/wp280/PhD/reykjanes/nodes/archive_wrong"
pattern = "*.2025.09.18.*.Z.miniseed"
component = "Z"
starttime = UTCDateTime("2025-09-18T19:05:00")
endtime   = UTCDateTime("2025-09-18T19:20:00")
event_lat = 0 # 51.636
event_lon = 0 # 160.017
filter_type = "bandpass"     # 'lowpass', 'highpass', 'bandpass'
freqmin = 0.5
freqmax = 1.5
# ----------------------

meta_nodes = pd.read_csv(csv_dir)

meta_nodes['event distance'] = [
    obspy.geodetics.base.gps2dist_azimuth(lat, lon, event_lat, event_lon)[0]
    for lat, lon in zip(meta_nodes['latitude'], meta_nodes['longitude'])
]

meta_nodes = meta_nodes.sort_values('event distance')
# print(meta_nodes)

# ----------------------

julday = starttime.strftime('%j')
year = starttime.strftime('%Y')

streams = []
inv = read_inventory("/raid2/wp280/PhD/reykjanes/nodes/dataless/xml/smartsolo_response.xml")

for _, row in meta_nodes.iterrows():
    if row['node_1'] == '-':
        pass
    pattern_node = f"{data_dir}/{year}/{julday}/*{row['code']}*{component}*"
    
    # adjust column name to match your file naming
    files = sorted(glob.glob(pattern_node))
    for f in files:
        try:
            st = read(f)
            st.trim(starttime=starttime, endtime=endtime)
            st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
            streams.append(st)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

#for i, st in enumerate(streams):
#    print(i, st[0].stats.station, st[0].stats.channel)

#for f in sorted(glob.glob(f"{data_dir}/{pattern}")):
#    try:
#        st = read(f)
#        st.trim(starttime=starttime, endtime=endtime)
#        st.filter(filter_type, freqmin=freqmin, freqmax=freqmax)
#        streams.append(st)
#    except Exception as e:
#        print(f"Failed to load {f}: {e}")

if not streams:
    raise SystemExit("No valid streams found.")

combined = Stream()
for st in streams:
	combined += st
combined._sort = False

combined.traces = [tr for st in streams for tr in st.traces]

#combined = sum(streams[1:], streams[0])
#combined.plot(equal_scale=False, size=(1000, 800), method="full", sort=None)

fig, axes = plt.subplots(len(streams), 1, figsize=(12, 10), sharex=True)

for i, st in enumerate(streams):
    tr = st[0]
    
    # tr.stats.network = "RK"
    # tr.stats.location = ""
    # tr.stats.station = meta_nodes[meta_nodes['node_1'] == str(tr.stats.station)]['code'].values[0]
    tr.remove_response(inventory=inv)
    # absolute times
    t_abs = [tr.stats.starttime.datetime + timedelta(seconds=t) for t in tr.times()]
    axes[i].plot(t_abs, tr.data, lw=0.8, color='black')
    axes[i].set_yticks([])
    axes[i].set_ylabel(tr.stats.station, rotation=0, labelpad=25)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axes[-1].set_xlabel("Time (UTC)")

plt.suptitle(f"Event at {event_lat}, {event_lon} around {starttime}")
plt.tight_layout()
plt.show()
