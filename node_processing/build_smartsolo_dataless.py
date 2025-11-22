# -*- coding: utf-8 -*-
"""
Build dataless response inventory for SmartSolo data (from stations.txt file
with coordinates etc lookup, plus LOG files containing sensitivity info).

Author: Tom Winder
Date: December 2023

Added options to build a dataless file for a replaced deployment
Added serial number information to each entry

Edited by Will Pizii, November 2025

"""

# imports
from pathlib import Path
import pandas as pd
from copy import deepcopy
import traceback, sys

from obspy import UTCDateTime
from obspy.clients.nrl import NRL
from obspy.core.inventory import (Inventory, Network, Station, Channel, Site,
                                  Equipment, Latitude, Longitude)

##############
# PARAMETERS #
##############

NET_CODE = "RK"
NET_DESCRIPTION = "Reykjanes nodes September-November 2025"
STATIONS_CSV = "/raid2/wp280/PhD/reykjanes/nodes/dataless/node_locations_times_nov.csv"

OUTPUT_XML = "/raid2/wp280/PhD/reykjanes/nodes/dataless/smartsolo_responseTEST.xml"

# If only one deployment, these should be strings. 
# If replaced, then they should be lists e.g. ['start_1', 'start_2']

start_name = ['start_1','start_2']
end_name = ['end_1', 'end_2']
serial = ['node_1', 'node_2']
loc_codes = ['00', '10']

# Number of deployments should match length of lists above, or 0 if not redeployed

deployments = 2

# Sensor parameters

preamp = 24             #    in dB
sampling_rate = 250     #    in Hz

# Source labelling string for dataless header

source_label = "Will Pizii, build_smartsolo_dataless.py, Nov 2025"

##############

preamp_lookup = {24:16, 30:32, 36:64}

nrl = NRL()
SMARTSOLO_RESP = nrl.get_response(sensor_keys=["DTCC (manuafacturers of SmartSolo)", "DT-SOLO", "5 Hz", "Rc=1850, Rs=430000"], 
                                  datalogger_keys=["DTCC (manufacturers of SmartSolo", "SmartSolo IGU-16HR3C", f"{preamp} dB ({preamp_lookup[preamp]})",
                                    f"{sampling_rate}", "Linear Phase", "Off"])

# Change NRL to match sensitivty quoted in the technical specs (as Zeckra)
SMARTSOLO_RESP.response_stages[0].stage_gain = 76.7
# Change NRL to correspond to miniSEED output in *counts*, at an ADC gain of
# 3355.4428 mV/count (as in technical specs, and Zeckra)
SMARTSOLO_RESP.response_stages[2].stage_gain *= 3355.4428

SMARTSOLO_RESP.recalculate_overall_sensitivity()

def parse_startdate(s):
    # if in format YYYY:DDD:HH:MM:SS
    if ":" in s and len(s.split(":")) == 5:
        year, doy, hour, minute, second = s.split(":")
        return UTCDateTime(year=int(year), julday=int(doy),
                           hour=int(hour), minute=int(minute), second=int(second))
    return UTCDateTime(s)

# lookups
COMP_AZ_LOOKUP = {
    "Z": 0.,
    "N": 0.,
    "E": 90.
}
COMP_DIP_LOOKUP = {
    "Z": -90.,
    "N": 0.,
    "E": 0.
}

station_df = pd.read_csv(STATIONS_CSV)

inv = Inventory(networks=[],
                source=source_label)

net = Network(code=NET_CODE, stations=[], description=NET_DESCRIPTION)

for _, station_row in station_df.iterrows():
    print(f"Working on {station_row.code}")

    try:
        station_start = next(v for v in (station_row[n] for n in start_name) if not pd.isna(v))
        station_end = next(v for v in (station_row[n] for n in end_name[::-1]) if not pd.isna(v))


        sta = Station(code=station_row.code,
                  latitude=Latitude(station_row.latitude),
                  longitude=Longitude(station_row.longitude),
                  elevation=station_row.elevation,
                  site=Site(name=station_row.description),
                  start_date=parse_startdate(station_start),
                  end_date=parse_startdate(station_end),
                  creation_date=UTCDateTime.now())
        
    except Exception as e:
        print("Exception type:", type(e).__name__)
        print("Exception message:", e)
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)
        break

    if deployments != 0:
        if len(start_name) != deployments or len(end_name) != deployments or len(serial) != deployments:
            raise ValueError(
                f"Mismatch: expected {deployments} entries, "
                f"got start_name={len(start_name)}, end_name={len(end_name)}, serial={len(serial)}"
            )

        repl_iter = zip(start_name, end_name, serial)
    else:
        repl_iter = [(start_name, end_name, serial)]

    for (start, end, sn), loc_code in zip(repl_iter, loc_codes):
        if sn == "-" or pd.isna(station_row[start]) or pd.isna(station_row[end]):
            continue

        for comp in "ZNE":
            cha = Channel(
                code=f"DP{comp}",
                location_code=loc_code,
                latitude=Latitude(station_row.latitude),
                longitude=Longitude(station_row.longitude),
                elevation=station_row.elevation,
                depth=0.,
                azimuth=COMP_AZ_LOOKUP[comp],
                dip=COMP_DIP_LOOKUP[comp],
                sample_rate=station_row.tap0,
                start_date=parse_startdate(station_row[start]),
                end_date=parse_startdate(station_row[end]),
                sensor=Equipment(type="SmartSolo IGU-16HR 3C 5 Hz",
                                serial_number=int(station_row[sn]))
            )
            cha.response = deepcopy(SMARTSOLO_RESP)
            sta.channels.append(cha)

            # Just taking generic sensitivity from the NRL file.
            # Though there is channel mapping info:
            # Ch1 = Z (Vertical)
            # Ch2 = X (North-South)
            # Ch3 = Y (East-West)
            resp = deepcopy(SMARTSOLO_RESP)

            cha.response = resp

            sta.channels.append(cha)

    net.stations.append(sta)

inv.networks.append(net)

inv.write(OUTPUT_XML, format="STATIONXML")
