import numpy as np
from pysurf96 import surf96
import matplotlib.pyplot as plt
import json, os

disp_json = ["/space/wp280/CCFRFR/TT_GROUP_NEW_PICKS.json", "/space/wp280/CCFRFR/TT_GROUP_PICKS.json"] # either path (for Rayleigh) or list of [Rayleigh, Love]
outfile = "model_dispersion_group.png"

model = None # 'SW Fit' # 'Weir' 'Jenkins' or 'All'
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
            1.45288843, 1.44507684, 1.62379505, 1.89368752, 1.99456806,
            2.21659218, 2.31062620, 2.39914717, 2.56575916, 2.61948276,
            2.72207439, 2.78094559, 2.88653910, 3.05656022, 3.13837114,
            3.22533815, 3.29019520, 3.32983607, 3.34503619, 3.35085603,
            3.34681162, 3.34833121, 3.32935858, 3.31371053, 3.28762516,
            3.27791505, 3.27310712, 3.26500182, 3.26655925, 3.28417574,
            3.30965255, 3.34207416, 3.39927314, 3.43809530, 3.47150879,
            3.50685983, 3.52845831, 3.53307322, 3.53946398, 3.54627562,
            3.54508167, 3.55004919, 3.53539124, 3.53440668, 3.52643928,
            3.52721161, 3.52503412, 3.52063594, 3.52595597, 3.51577359,
            3.51067485, 3.50688798, 3.50540929, 3.50153431, 3.50387358,
            3.50449907, 3.50690125, 3.50387917, 3.49366891, 3.48402147,
            3.47877810
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
    if not plot_all
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
