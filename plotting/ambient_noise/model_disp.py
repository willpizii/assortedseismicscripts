import numpy as np
from pysurf96 import surf96
import matplotlib.pyplot as plt
import json, os

disp_json = ["/space/wp280/CCFRFR/ZZ_GROUP_NEW_PICKS.json", "/space/wp280/CCFRFR/TT_GROUP_NEW_PICKS.json"] # either path (for Rayleigh) or list of [Rayleigh, Love]
outfile = "model_dispersion_group.png"

model = 'BBInv' # 'SW Fit' # 'Weir' 'Jenkins' or 'All'
wavetype = 'both' # 'rayleigh', 'love' or 'both'
veltype = 'group'
plot_all = False

fig, ax = plt.subplots(figsize=[10,8])

def plot_model(model):
    if model == 'Jenkins':  # Jenkins et al 2025
        thickness = np.array([1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,1.0,1.0,1.0,
                            10.0,0.0])
        vs =        np.array([2.4,2.8,3.1,3.4,3.5,
                            3.6,3.6,3.7,3.8,3.9,
                            4.0,4.0,4.1,4.1,4.1,
                            4.1,4.2])
        vp = vs * 1.73

    elif model == 'Weir':   # Weir et al 2001
        thickness = np.array([1.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,2.0,2.0,2.0,
                            3.0,3.0,0.0])
        vp =        np.array([2.5,3.7,4.7,5.8,6.3,
                            6.7,6.8,6.9,7.0,7.1,
                            7.2,7.7,7.8])
        vs = vp / 1.73

    elif model == 'Allas':
        thickness = np.array([
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.5,
            0.5,0.5,0.5,0.5,0.0
        ])

        vp = np.array([
            2.4583,2.3322,2.3036,2.9745,4.3255,
            5.0042,5.414,5.8182,6.1942,6.4598,
            6.6401,6.7223,6.8175,6.8207,6.8268,
            6.825,6.8163,6.8095,6.8084,6.8428,
            6.8763,6.9081,6.9391,6.9734,7.01,
            7.0682,7.1298,7.1906,7.2529,7.3144,
            7.376,7.6789,7.6958,7.7137,7.7313
        ])

        vs = np.array([
            1.6296,1.5495,1.5242,1.8677,2.5425,
            2.8329,3.0366,3.2792,3.5536,3.6985,
            3.7315,3.745,3.7873,3.748,3.7106,
            3.6864,3.6677,3.6512,3.6526,3.6978,
            3.7587,3.8212,3.8756,3.9194,3.9543,
            3.9944,4.0304,4.0644,4.0982,4.1324,
            4.167,4.3388,4.3486,4.3585,4.3679
        ])

    elif model == 'Southern':
        thickness = np.array([
            0.75,0.25,0.25,0.25,0.25,
            0.25,0.25,0.25,0.25,0.25,
            0.25,0.25,0.25,0.25,0.25,
            0.25,0.25,0.25,0.25,0.25,
            0.25,0.25,3.0,3.0,3.0,
            0.1,4.9,
            0.0
        ])

        vp = np.array([
            2.336,2.529,2.546,2.745,3.176,
            3.778,4.419,4.842,5.047,5.283,
            5.471,5.688,5.898,6.086,6.280,
            6.378,6.470,6.514,6.561,6.566,
            6.624,6.681,6.723,6.764,7.008,
            7.376,7.665,7.835
        ])

        vs = np.array([
            1.320,1.429,1.438,1.551,1.794,
            2.134,2.497,2.736,2.851,2.985,
            3.091,3.214,3.332,3.438,3.548,
            3.603,3.655,3.680,3.707,3.710,
            3.742,3.775,3.798,3.821,3.959,
            4.167,4.331,4.427
        ])

    elif model == 'SW Fit':
        thickness = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
        vs        = np.array([1.5,1.8,2.1,2.4,2.7,3.0,3.1,3.2,3.3,3.3,3.3,3.3,3.3,3.4,3.4,3.4,3.4,3.5,3.5,3.5,3.5,3.6])
        vp        = vs * 1.73

    elif model == 'BBInv':
        thickness = np.array([
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25
        ])

        vs = np.array([
            1.63653782, 1.64615589, 1.66804864, 1.72184135, 2.06447057, 2.29294483,
            2.36796888, 2.46204454, 2.60902126, 2.76289003, 2.89202129, 2.99760202,
            3.11476658, 3.20120389, 3.28089564, 3.34735057, 3.3945541,  3.43396244,
            3.4709179,  3.5022915,  3.53257819, 3.55869297, 3.58601256, 3.60911273,
            3.62962052, 3.65087224, 3.67606988, 3.69827861, 3.71438812, 3.73119269,
            3.74897988, 3.76602505, 3.78428013, 3.79754497, 3.81192499, 3.82133772,
            3.83047812, 3.83399974, 3.83692072, 3.83987433, 3.83746153, 3.83499163,
            3.82983894, 3.81747652, 3.80345512, 3.79188893, 3.77988362, 3.76846859,
            3.74824006, 3.73310745, 3.72033887, 3.7207824,  3.72665581, 3.74282116,
            3.78101345, 3.82594678, 3.88929258, 3.9645195,  4.04875731, 4.12574474,
            4.15809137
        ])
        vp = vs * 1.73


    rho = vp * 0.32 + 0.77

    # Periods we are interested in
    periods = np.linspace(1.0, 10.0, 20)

    if wavetype == "both":
        velocities = surf96(
            thickness,
            vp,
            vs,
            rho,
            periods,
            wave='love',
            mode=1,
            velocity=veltype,
            flat_earth=True)
        
        ax.plot(periods,velocities, label=model+' love', color=f"C{j}")

        velocities = surf96(
            thickness,
            vp,
            vs,
            rho,
            periods,
            wave="rayleigh",
            mode=1,
            velocity=veltype,
            flat_earth=True)
        
        ax.plot(periods,velocities, label=model+ ' rayleigh', color=f"C{j}", ls="--")

    else:
        velocities = surf96(
            thickness,
            vp,
            vs,
            rho,
            periods,
            wave=wavetype,
            mode=1,
            velocity=veltype,
            flat_earth=True)
        
        ax.plot(periods,velocities, label=model, color=f"C{j}")

if model == 'All':
    j=0
    for m in ['Weir', 'Jenkins', 'Allas', 'Southern', 'SW Fit', 'BBInv']:
        plot_model(m)
        j+=1

elif model == None:
    j=1

else:
    j=0
    plot_model(model)

if disp_json and type(disp_json) == str:
    with open(disp_json, "r") as f:
        ridge_dict = json.load(f)
    

    all_p = np.unique(np.concatenate([np.array(v[0], float) for v in ridge_dict.values()]))
    vals = {p: [] for p in all_p}

    for key, data in ridge_dict.items():
        p = np.array(data[0], dtype=float)
        v = np.array(data[1], dtype=float)
        if plot_all:
            ax.plot(p, v, lw=1, color=f"C{j}", alpha=0.1)

        for pi, vi in zip(p, v):
            vals[pi].append(vi)

    mean_periods = []
    mean_vals = []
    std_vals = []

    for p in all_p:
        if len(vals[p]) > 0:
            mean_periods.append(p)
            mean_vals.append(np.mean(vals[p]))
            std_vals.append(np.std(vals[p]))

    mean_periods = np.array(mean_periods)
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    ax.plot(mean_periods, mean_vals, lw=2, color=f"C{j}")
    if not plot_all:
        ax.fill_between(mean_periods, mean_vals - std_vals, mean_vals + std_vals, color=f"C{j}", alpha=0.5)

elif disp_json and type(disp_json) == list:

    for dj, ls in zip(disp_json, ['--','-']):
        j+=1
        with open(dj, "r") as f:
            ridge_dict = json.load(f)
        

        all_p = np.unique(np.concatenate([np.array(v[0], float) for v in ridge_dict.values()]))
        vals = {p: [] for p in all_p}

        for key, data in ridge_dict.items():
            p = np.array(data[0], dtype=float)
            v = np.array(data[1], dtype=float)

            for pi, vi in zip(p, v):
                vals[pi].append(vi)

        mean_periods = []
        mean_vals = []
        std_vals = []

        for p in all_p:
            if len(vals[p]) > 0:
                mean_periods.append(p)
                mean_vals.append(np.mean(vals[p]))
                std_vals.append(np.std(vals[p]))

        mean_periods = np.array(mean_periods)
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)

        ax.plot(mean_periods, mean_vals, lw=2, color=f"C{j}", ls=ls,label=os.path.basename(dj).split(".")[0])
        ax.fill_between(mean_periods, mean_vals - std_vals, mean_vals + std_vals, color=f"C{j}", alpha=0.5)


ax.legend()
ax.set_xlabel("Period (s)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Loaded Dispersion Curves")

if outfile:
    plt.savefig(outfile)

plt.show()
