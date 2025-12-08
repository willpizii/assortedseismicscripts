from obspy import read
import glob, os

parent_dir = '/raid2/wp280/PhD/reykjanes/nodes/msnoise-main/robust/EGF/ZZ'

files = glob.glob(os.path.join(parent_dir,"*FISH*VIGR*"))

for f in files:
    st = read(f)
    st[0].data *= -1
    st.write(f, format='MSEED')