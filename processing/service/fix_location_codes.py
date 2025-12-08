from obspy import read
import glob, os
from tqdm import tqdm

path = '/raid4/Iceland/reykjanes_data/raw_data/36_november_2025/service_data/'
stat = 'LANG'
data = ['velocity', 'masses', 'environmental']
year = '2025'
netw = 'RK'
loca = '41'

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
			
			