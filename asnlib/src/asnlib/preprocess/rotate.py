import numpy as np, obspy, glob, os
from math import sin, cos
from obspy import read
try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

def get_az_baz(x1,y1,x2,y2, radians=True):
    _, az, baz = obspy.geodetics.base.gps2dist_azimuth(y1,x1,y2,x2)
    if radians:
        return(az * np.pi / 180, baz * np.pi / 180)
    else:
        return(az,baz)

def pairwise_rotation_matrix(az,baz):
    rot_matrix = np.array([[-cos(az) * cos(baz), cos(az) * sin(baz), -sin(az) * sin(baz), sin(az) * cos(baz)],
                  [-sin(az) * sin(baz), -sin(az) * cos(baz), -cos(az) * cos(baz), -cos(az) * sin(baz)],
                  [-cos(az) * sin(baz), -cos(az) * cos(baz), sin(az) * cos(baz), sin(az) * sin(baz)],
                  [-sin(az) * cos(baz), sin(az)  * sin(baz), cos(az) * sin(baz), -cos(az) * cos(baz)]])
    return(rot_matrix)

def find_component_files(cc_dir, pattern, components=["EE","EN","NN","NE"]):
    # find all files
    files = {}
    
    # search and check for components
    for c in components:
        if not os.path.exists(os.path.join(cc_dir, c, pattern)):
            # tqdm.write(f"Missing component {c} for {pattern}, skipping...")
            return
        files[c] = sorted(glob.glob(os.path.join(cc_dir, c, pattern,"*")))

    return(files)

def load_components(EE_path, EN_path, NN_path, NE_path):
    EE = read(EE_path)[0].data
    EN = read(EN_path)[0].data
    NN = read(NN_path)[0].data
    NE = read(NE_path)[0].data

    return(EE,EN,NN,NE)

def rotate_rr_tt(EE,EN,NN,NE,rot_matrix):
    # define data vector and transformed rotated vector
    data_vector = np.vstack([EE,EN,NN,NE])
    rot_vector = rot_matrix @ data_vector

    return(rot_vector)