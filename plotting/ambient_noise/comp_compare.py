import json, os
import numpy as np
import matplotlib.pyplot as plt

disp_files = ["/space/wp280/CCFRFR/ZZ_PICKS.json",
              "/space/wp280/CCFRFR/RR_PICKS.json",
              "/space/wp280/CCFRFR/TT_PICKS.json"]

fig, ax = plt.subplots(figsize=[10,8])

j=0
for disp_json in disp_files:
    with open(disp_json, "r") as f:
        ridge_dict = json.load(f)
    

    all_p = np.unique(np.concatenate([np.array(v[0], float) for v in ridge_dict.values()]))
    vals = {p: [] for p in all_p}

    for key, data in ridge_dict.items():
        p = np.array(data[0], dtype=float)
        v = np.array(data[1], dtype=float)

        ax.plot(p, v, lw=1, color=f'C{j}', alpha=0.1,zorder=0)

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

    ax.plot(mean_periods, mean_vals, lw=2, color=f'C{j}', label=os.path.basename(disp_json).split(".")[0])
    ax.fill_between(mean_periods, mean_vals - std_vals, mean_vals + std_vals, color=f'C{j}', alpha=0.5)
    j+=1

ax.set_xlabel("Period (s)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
plt.show()