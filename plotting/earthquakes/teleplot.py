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
data_dir = "/raid2/wp280/PhD/reykjanes/nodes/archive_wrong/RK"
component = "Z"
filter_type = "bandpass"     # 'lowpass', 'highpass', 'bandpass'
freqmin = 0.5
freqmax = 2.0

starttime = UTCDateTime("2025-09-18T19:05:00")
endtime   = UTCDateTime("2025-09-18T19:20:00")
event_lat = 0 # 51.636
event_lon = 0 # 160.017
# ----------------------

inv = read_inventory("/raid2/wp280/PhD/reykjanes/nodes/dataless/xml/smartsolo_response.xml")
catalogue = pd.read_csv("/raid2/wp280/PhD/reykjanes/nodes/plot/catalogue.csv")

# ----------------------

def on_key(event):
    if event.key == "y":
        print(f"accepted (Mw {magnitude})")
    elif event.key == "n":
        print(f"rejected (Mw {magnitude})")
    plt.close(fig)

meta_nodes = pd.read_csv(csv_dir)

print(meta_nodes)

meta_nodes['event distance'] = [
    obspy.geodetics.base.gps2dist_azimuth(lat, lon, event_lat, event_lon)[0]
    for lat, lon in zip(meta_nodes['latitude'], meta_nodes['longitude'])
]

meta_nodes = meta_nodes.sort_values('event distance')

# ----------------------

julday = starttime.strftime('%j')
year = starttime.strftime('%Y')

streams = []

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

if not streams:
    raise SystemExit("No valid streams found.")

fig, axes = plt.subplots(len(streams), 1, figsize=(12, 10), sharex=True)

for i, st in enumerate(streams):
    if len(st) == 0:
        print("WARNING: missing trace. Details:")
        print(st)

        continue

    tr = st[0]
    tr.remove_response(inventory=inv)

    # absolute times
    t_abs = [tr.stats.starttime.datetime + timedelta(seconds=t) for t in tr.times()]
    axes[i].plot(t_abs, tr.data, lw=0.8, color='black')
    axes[i].set_yticks([])
    axes[i].set_ylabel(tr.stats.station, rotation=0, labelpad=25)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axes[-1].set_xlabel("Time (UTC)")

#plt.suptitle(f"Mw {magnitude} event at {event_lat}, {event_lon} around {starttime}")
plt.tight_layout()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

    
