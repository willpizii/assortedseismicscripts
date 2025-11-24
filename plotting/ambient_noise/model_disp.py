import numpy as np
from pysurf96 import surf96
import matplotlib.pyplot as plt

# Define the velocity model in km and km/s
thickness = np.array([1.0,1.0,1.0,1.0,1.0,
                      1.0,1.0,1.0,1.0,1.0,
                      1.0,1.0,1.0,1.0,1.0,
                      10.0,0.0])
vs =        np.array([2.4,2.8,3.1,3.4,3.5,
                      3.6,3.6,3.7,3.8,3.9,
                      4.0,4.0,4.1,4.1,4.1,
                      4.1,4.2])
vp = vs * 1.73
rho = vp * 0.32 + 0.77

# Periods we are interested in
periods = np.linspace(1.0, 10.0, 20)

velocities = surf96(
    thickness,
    vp,
    vs,
    rho,
    periods,
    wave="love",
    mode=1,
    velocity="phase",
    flat_earth=True)

plt.plot(periods,velocities)
plt.show()