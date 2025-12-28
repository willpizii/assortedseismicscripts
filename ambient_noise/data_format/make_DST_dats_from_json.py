import os, json, pandas as pd

stations = pd.read_csv('/space/wp280/CCFRFR/frfr_stations.csv')
json_pat = '/space/wp280/CCFRFR/ZZ_PICKS.json'
out_path = '/space/wp280/DAST/ReykjanesRawData'

with open(json_pat, 'r') as f:
    data = json.load(f)

for key, value in data.items():
    sta1 = key.split('_')[-3]
    sta2 = key.split('_')[-1]

    sta1d = stations[stations['sta'] == sta1]
    sta2d = stations[stations['sta'] == sta2]

    outfile = os.path.join(out_path, 'CDisp.'+sta1+'-'+sta2+'.dat')

    with open(outfile, 'w') as f:
        f.write(str(sta1d.X.iloc[0]) + " " + str(sta1d.Y.iloc[0]) + '\n')
        f.write(str(sta2d.X.iloc[0]) + " " + str(sta2d.Y.iloc[0]) + '\n')

        for k, v in zip(value[0], value[1]):
            f.write(str(round(k, 4)) + "  " + str(round(v, 4)) + '\n')
