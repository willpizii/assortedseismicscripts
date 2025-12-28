import os, json, pandas as pd, numpy as np

############
# PARAMETERS
############

ignore_stations = ['LAMB', 'SMAL', 'THOR']

stations = pd.read_csv('/space/wp280/CCFRFR/frfr_stations.csv')

rayleigh_pat = '/space/wp280/CCFRFR/ZZ_PICKS.json'
love_pat = '/space/wp280/CCFRFR/TT_PICKS.json' 

out_path = '/space/wp280/DAST/ReykjanesRawData'

############

with open(rayleigh_pat, 'r') as f:
    rayleigh_data = json.load(f)

with open(love_pat, 'r') as f:
    love_data = json.load(f)

def keep_key(key):
    parts = key.split('_')
    return not any(sta in parts for sta in ignore_stations)

rayleigh_data = {
    k: v for k, v in rayleigh_data.items()
    if keep_key(k)
}

love_data = {
    k: v for k, v in love_data.items()
    if keep_key(k)
}

############
# RAYLEIGH #
############

sta1latall = []
sta1lonall = []
sta2latall = []
sta2lonall = []
periodsall = []
dispersionall = []
stationpairID = []

for key, value in rayleigh_data.items():
    sta1 = key.split('_')[-3]
    sta2 = key.split('_')[-1]

    sta1d = stations[stations['sta'] == sta1]
    sta2d = stations[stations['sta'] == sta2]

    pairid = sta1+'-'+sta2

    sta1lon = sta1d.X.iloc[0]
    sta1lat = sta1d.Y.iloc[0]
    sta2lon = sta2d.X.iloc[0]
    sta2lat = sta2d.Y.iloc[0]

    _,nperiods = np.array(value).shape
    
    periods = np.zeros(nperiods,)
    disper  = np.zeros(nperiods,)

    for ii in range(nperiods):
        periods[ii] = np.array(value)[0,ii]
        disper[ii]  = round(np.array(value)[1,ii],5)
    
    dispersionall = np.hstack([dispersionall, disper])
    periodsall = np.hstack([periodsall, periods])

    sta1latall = np.hstack([sta1latall,np.ones(nperiods,)*sta1lat])
    sta1lonall = np.hstack([sta1lonall,np.ones(nperiods,)*sta1lon])
    sta2latall = np.hstack([sta2latall,np.ones(nperiods,)*sta2lat])
    sta2lonall = np.hstack([sta2lonall,np.ones(nperiods,)*sta2lon])
    stationpairID = stationpairID + [pairid] * nperiods

dataall = pd.DataFrame({'sta1lat':sta1latall, 'sta1lon':sta1lonall, \
                        'sta2lat':sta2latall, 'sta2lon':sta2lonall, \
                        'periods':periodsall, 'dispersion': dispersionall, \
                        'pairid':stationpairID})

dataall = dataall.sort_values(by = ['periods', 'pairid'])
dataall = dataall.set_index('periods')

wavetype = 2
phasetype = 0

with open('surfdata.dat','w') as fout:
    UniqPeriods = dataall.index.unique()
    for iperiod,period in enumerate(UniqPeriods):
        datasubperiod = dataall.loc[period]
        if isinstance(datasubperiod,pd.Series):
            continue
        datasubperiod = datasubperiod.reset_index().set_index('sta1lat')
        sta1lat = datasubperiod.index.unique()
        for ista1 in sta1lat:
            datasubstation = datasubperiod.loc[ista1]
            if isinstance(datasubstation,pd.DataFrame):
                fout.write(f'# {datasubstation.index[0]} {datasubstation["sta1lon"].iloc[0]} {iperiod+1} {wavetype} {phasetype}\n')
                for ista2 in range(len(datasubstation)):
                    fout.write(f'{datasubstation["sta2lat"].iloc[ista2]} {datasubstation["sta2lon"].iloc[ista2]} {datasubstation["dispersion"].iloc[ista2]}\n')
            else:
                fout.write(f'# {datasubstation.name} {datasubstation["sta1lon"]} {iperiod+1} {wavetype} {phasetype}\n')
                fout.write(f'{datasubstation["sta2lat"]} {datasubstation["sta2lon"]} {datasubstation["dispersion"]}\n')

########
# LOVE #
########

sta1latall = []
sta1lonall = []
sta2latall = []
sta2lonall = []
periodsall = []
dispersionall = []
stationpairID = []

for key, value in love_data.items():
    sta1 = key.split('_')[-3]
    sta2 = key.split('_')[-1]

    sta1d = stations[stations['sta'] == sta1]
    sta2d = stations[stations['sta'] == sta2]

    pairid = sta1+'-'+sta2

    sta1lon = sta1d.X.iloc[0]
    sta1lat = sta1d.Y.iloc[0]
    sta2lon = sta2d.X.iloc[0]
    sta2lat = sta2d.Y.iloc[0]

    _,nperiods = np.array(value).shape
    
    periods = np.zeros(nperiods,)
    disper  = np.zeros(nperiods,)

    for ii in range(nperiods):
        periods[ii] = np.array(value)[0,ii]
        disper[ii]  = round(np.array(value)[1,ii],5)
    
    dispersionall = np.hstack([dispersionall, disper])
    periodsall = np.hstack([periodsall, periods])

    sta1latall = np.hstack([sta1latall,np.ones(nperiods,)*sta1lat])
    sta1lonall = np.hstack([sta1lonall,np.ones(nperiods,)*sta1lon])
    sta2latall = np.hstack([sta2latall,np.ones(nperiods,)*sta2lat])
    sta2lonall = np.hstack([sta2lonall,np.ones(nperiods,)*sta2lon])
    stationpairID = stationpairID + [pairid] * nperiods

dataall = pd.DataFrame({'sta1lat':sta1latall, 'sta1lon':sta1lonall, \
                        'sta2lat':sta2latall, 'sta2lon':sta2lonall, \
                        'periods':periodsall, 'dispersion': dispersionall, \
                        'pairid':stationpairID})

dataall = dataall.sort_values(by = ['periods', 'pairid'])
dataall = dataall.set_index('periods')

wavetype = 1
phasetype = 0

with open('surfdata.dat','a') as fout:
    UniqPeriods = dataall.index.unique()
    for iperiod,period in enumerate(UniqPeriods):
        datasubperiod = dataall.loc[period]
        if isinstance(datasubperiod,pd.Series):
            continue
        datasubperiod = datasubperiod.reset_index().set_index('sta1lat')
        sta1lat = datasubperiod.index.unique()
        for ista1 in sta1lat:
            datasubstation = datasubperiod.loc[ista1]
            if isinstance(datasubstation,pd.DataFrame):
                fout.write(f'# {datasubstation.index[0]} {datasubstation["sta1lon"].iloc[0]} {iperiod+1} {wavetype} {phasetype}\n')
                for ista2 in range(len(datasubstation)):
                    fout.write(f'{datasubstation["sta2lat"].iloc[ista2]} {datasubstation["sta2lon"].iloc[ista2]} {datasubstation["dispersion"].iloc[ista2]}\n')
            else:
                fout.write(f'# {datasubstation.name} {datasubstation["sta1lon"]} {iperiod+1} {wavetype} {phasetype}\n')
                fout.write(f'{datasubstation["sta2lat"]} {datasubstation["sta2lon"]} {datasubstation["dispersion"]}\n')


print('Finished reformatting dispersion data')
