import pandas as pd

inp = ["/home/wp280/Downloads/Network Aug25 - Oct25 - Sheet1.csv",
       "//home/wp280/Downloads/Network Oct25-Nov25 - Sheet1.csv"]  # 'a' or ['a', 'b']
out_cert = "/raid4/Iceland/reykjanes_data/dataless/services/Aug25-Nov25/CERT_stations.txt"
out_rest = "/raid4/Iceland/reykjanes_data/dataless/services/Aug25-Nov25/stations.txt"

if isinstance(inp, str):
    df = pd.read_csv(inp)

elif isinstance(inp, list):
    df_list = [pd.read_csv(c) for c in inp]
    df = pd.concat(df_list, ignore_index=True)

instrument_map = {
    'Certimus':'CERT2',
    '3ESPCD':'3ESP',
    '16GB 6TD':'6T',
    '4GB 6TD':'6t'
}

df['Instrument'] = df['Instrument'].map(instrument_map).fillna(df['Instrument'])

df['Location Code'] = df['Location Code'].str.replace(' ', '', regex=False)
df['Serial Number'] = df['Serial Number'].str.replace(' ', '', regex=False)

cert_df = df[df['Instrument'] == 'CERT2']
oth_df  = df[df['Instrument'] != 'CERT2']

# Certimus

cols = ['st_d', 'en_d', 'tap2', 'tapcode', 'notes']
values = ['FILLMEIN', 'FILLMEIN', '1', '???', 'FILLMEIN']

cert_df = cert_df.copy()

for col, val in zip(cols, values):
    cert_df[col] = val

c_df = cert_df[['Station Code', 'Instrument',
                     'Serial Number', 'Firmware Version',
                     'Femtomus Firmware', 'Latitude', 'Longitude',
                     'Elevation / m', 'st_d', 'en_d', 'Sample Rate',
                     'tap2', 'tapcode','Long Name', 'Location Code', 'notes']]

c_df = c_df.drop_duplicates()
c_df = c_df.sort_values(['Station Code', 'Location Code'])

c_df['Elevation / m'] = c_df['Elevation / m'].astype(int)
c_df['Sample Rate'] = c_df['Sample Rate'].astype(int)

groups = []
cols = c_df.columns

for _, g in c_df.groupby('Station Code'):
    groups.append(g)
    groups.append(pd.DataFrame([[''] * len(cols)], columns=cols))

out = pd.concat(groups, ignore_index=True)

w = {
    'Station Code': 5,
    'Instrument': 6,
    'Serial Number': 6,
    'Firmware Version': 12,
    'Femtomus Firmware': 12,
    'Latitude': 10,
    'Longitude': 10,
    'Elevation / m': 4,
    'st_d': 10,
    'en_d': 10,
    'Sample Rate': 4,
    'tap2': 4,
    'tapcode': 4,
    'Long Name': 24,
    'Location Code': 3,
    'notes': 20
}

lines = []

for _, row in out[w.keys()].iterrows():
    pieces = [
        str(row[col])[: w[col]].ljust(w[col])
        for col in w.keys()
    ]
    lines.append(''.join(pieces))

with open(out_cert, 'w') as f:
    f.write('\n'.join(lines))

# Other instruments

cols = ['dig', 'st_d', 'en_d', 'tap2', 'tapcode', 'notes']
values = ['DMCD24', 'FILLMEIN', 'FILLMEIN', 'X', '1XX', 'FILLMEIN']

oth_df = oth_df.copy()

for col, val in zip(cols, values):
    oth_df[col] = val

o_df = oth_df[['Station Code', 'Instrument',
                     'Serial Number', 'dig', 'Latitude', 'Longitude',
                     'Elevation / m', 'st_d', 'en_d', 'Sample Rate',
                     'tap2', 'tapcode','Long Name', 'Location Code', 'notes']]

o_df = o_df.drop_duplicates().dropna(subset="Station Code")
o_df = o_df.sort_values(['Station Code', 'Location Code'])

o_df['Elevation / m'] = o_df['Elevation / m'].astype(int)
o_df['Sample Rate'] = o_df['Sample Rate'].astype(int)

tmp = o_df['Serial Number'].str.split(pat='-', n=1, expand=True)

tmp = tmp.reindex(columns=[0,1])

tmp[1] = tmp[1].fillna('')

o_df['sn1'] = tmp[0]
o_df['sn2'] = tmp[1]

groups = []
cols = o_df.columns

for _, g in o_df.groupby('Station Code'):
    groups.append(g)
    groups.append(pd.DataFrame([[''] * len(cols)], columns=cols))

out = pd.concat(groups, ignore_index=True)

w = {
    'Station Code': 5,
    'Instrument': 6,
    'sn1': 8,
    'dig':8,
    'sn2':6,
    'Latitude': 10,
    'Longitude': 10,
    'Elevation / m': 4,
    'st_d': 10,
    'en_d': 10,
    'Sample Rate': 4,
    'tap2': 4,
    'tapcode': 4,
    'Long Name': 24,
    'Location Code': 3,
    'notes': 20
}

lines = []

for _, row in out[w.keys()].iterrows():
    pieces = [
        str(row[col])[: w[col]].ljust(w[col])
        for col in w.keys()
    ]
    lines.append(''.join(pieces))

with open(out_rest, 'w') as f:
    f.write('\n'.join(lines))


