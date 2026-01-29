from obspy import read
import glob, os
from tqdm import tqdm

path = '/mnt/cambridge/indus/raid4/Iceland/reykjanes_data/raw_data'
stat = 'LYNG'
data = ['NODES_TEMP'] # ['velocity', 'masses', 'environmental']
year = '2025'
netw = 'RK'
loca = '10'

remove = True

for d in data:
	print(d.split('/')[-1])
	channels = glob.glob(os.path.join(path, d, year, netw, stat,'*'))
	for c in channels:
		print(c.split('/')[-1])
		mseeds = glob.glob(os.path.join(c, "*"))
		for m in tqdm(mseeds):
			st = read(m)
			for tr in st:
				tr.stats.location = loca
			outp = '.'.join(['.'.join(m.split('.')[:-5]),loca,'.'.join(m.split('.')[-4:])])
			if remove:
				os.remove(m)	
			st.write(outp, format='MSEED')
			
			
