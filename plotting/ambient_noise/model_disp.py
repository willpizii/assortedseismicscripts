import numpy as np
from pysurf96 import surf96
import matplotlib.pyplot as plt

model = 'All' # 'Weir' 'Jenkins' or 'All'

fig, ax = plt.subplots()

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
        thickness = np.array([])
        vp =        np.array([])
        vs =        np.array([])

        pass

    rho = vp * 0.32 + 0.77

    # Periods we are interested in
    periods = np.linspace(1.0, 10.0, 20)

    velocities = surf96(
        thickness,
        vp,
        vs,
        rho,
        periods,
        wave="rayleigh",
        mode=1,
        velocity="phase",
        flat_earth=True)
    
    ax.plot(periods,velocities, label=model)


if model == 'All':
    for m in ['Weir', 'Jenkins']:
        plot_model(m)
    ax.legend()

else:
    plot_model(model)

plt.show()