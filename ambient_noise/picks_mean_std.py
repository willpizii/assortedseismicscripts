import csv
import numpy as np
import json

disp_json = "/raid2/wp280/PhD/reykjanes/nodes/msnoise-main/picked_ridges_DEP.json"

with open(disp_json, "r") as f:
    ridge_dict = json.load(f)

all_p = np.unique(np.concatenate([np.array(v[0], float) for v in ridge_dict.values()]))
vals = {p: [] for p in all_p}

for data in ridge_dict.values():
    p = np.array(data[0], float)
    v = np.array(data[1], float)
    for pi, vi in zip(p, v):
        vals[pi].append(vi)

rows = []
for p in all_p:
    if vals[p]:
        m = np.mean(vals[p])
        s = np.std(vals[p])
        rows.append((p, m, s))

rows.sort(key=lambda x: x[0])

with open("data.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(["Frequency", "Velocity", "Velstd"])
    for r in rows:
        w.writerow(r)