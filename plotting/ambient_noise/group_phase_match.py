import json, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score

##############

target_period = 2.3

phase_json = "/space/wp280/CCFRFR/ZZ_PICKS.json"
group_json = "/space/wp280/CCFRFR/ZZ_GROUP_PICKS.json"

outfile = "match.png"

##############

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

with open(phase_json, "r") as f:
    phase_dict = json.load(f)

all_phase = np.unique(np.concatenate([np.array(v[0], float) for v in phase_dict.values()]))

phase_period = find_nearest(all_phase, target_period)

with open(group_json, "r") as f:
    group_dict = json.load(f)

all_group = np.unique(np.concatenate([np.array(v[0], float) for v in group_dict.values()]))

group_period = find_nearest(all_group, phase_period)

fig, ax = plt.subplots()

x_data, y_data = [], []

for key, value in phase_dict.items():

    if phase_period not in value[0]:
        continue

    if key not in group_dict:
        continue

    group = group_dict[key]

    if group_period not in group[0]:
        continue

    phase_velocity = value[1][value[0].index(phase_period)]
    group_velocity = group[1][group[0].index(group_period)]

    x_data.append(phase_velocity)
    y_data.append(group_velocity)

if len(x_data) > 2:
    # Reshape for scikit-learn
    X = np.array(x_data).reshape(-1, 1)
    y = np.array(y_data)

    # Initialize and fit RANSAC
    ransac = RANSACRegressor(residual_threshold=0.33, random_state=12345)
    ransac.fit(X, y)

    # Get the inlier mask (True for clean data, False for outliers)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Generate points for the trendline
    line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y_ransac = ransac.predict(line_X)

    # Plotting
    # 1. Plot the actual trendline
    ax.plot(line_X, line_y_ransac, color='green', linewidth=2, label='Robust Trendline')

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    inlier_preds = ransac.predict(X[inlier_mask])
    r2_clean = r2_score(y[inlier_mask], inlier_preds)

    print("--- Robust Regression Statistics ---")
    print(f"Model Equation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"Slope:          {slope:.4f}")
    print(f"Intercept:      {intercept:.4f}")
    print(f"R² (Inliers):   {r2_clean:.4f}")
    print(f"Inliers found:  {sum(inlier_mask)} / {len(x_data)} points")
    print(f"Outliers:       {sum(outlier_mask)} points")

ax.scatter(x_data, y_data)

plt.savefig(outfile)
plt.show()

results = []

# Loop through every unique period available in the phase dataset
for p_target in all_phase:
    x_p, y_p = [], []
    
    # 2. For THIS period, find the global 'nearest' group period
    # (Matches your first-half logic)
    g_target = find_nearest(all_group, p_target)
    
    for key, phase_data in phase_dict.items():
        # Check if station exists in both
        if key not in group_dict:
            continue
        group_data = group_dict[key]

        if p_target not in phase_data[0]:
            continue
        p_station = p_target
        if g_target not in group_data[0]:
            continue
        g_station = g_target

        phase_vel = phase_data[1][phase_data[0].index(p_station)]
        group_vel = group_data[1][group_data[0].index(g_station)]

        x_p.append(phase_vel)
        y_p.append(group_vel)
    
    # Only fit if we have enough points for a statistical trend
    if len(x_p) > 10:
        X = np.array(x_p).reshape(-1, 1)
        y = np.array(y_p)
        
        # RANSAC handles the 'miles away' points automatically
        # residual_threshold=0.2 is a good middle ground for km/s
        reg = RANSACRegressor(residual_threshold=0.33, random_state=12345)
        reg.fit(X, y)
        
        slope = reg.estimator_.coef_[0]
        inlier_count = sum(reg.inlier_mask_)
        
        results.append([p_target, slope, inlier_count])

# Convert to array for easy plotting
results = np.array(results)

# Plotting the Trend of Slopes
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results[:, 0], results[:, 1], 'ko-', label='Robust Slope (U/c trend)')
ax.axhline(1.0, color='r', linestyle='--', label='No Dispersion (U=c)')
ax.set_xlabel('Period (s)')
ax.set_ylabel('Regression Slope')
ax.set_title('Seismic Dispersion Slope across Periods')
ax.grid(True, alpha=0.3)
ax.legend()

plt.savefig("slope_vs_period.png")
plt.show()