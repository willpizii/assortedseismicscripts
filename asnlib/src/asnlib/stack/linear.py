import numpy as np

# As described by Pavlis and Vernon, 2010
# Shown by Yang et al. 2023 to be good for phase dispersion and other analysis

def linear_stack(ncfs):
    """
    Takes given signals, such as ncfs, and returns the linear stack
    
    Parameters:
    ncfs: np.stack of traces to process; in stack of N traces x M samples

    Returns:
    b: final linear stack
    """

    # cast input trace stack to array
    ncfs = np.asarray(ncfs, dtype=float)

    # normalise amplitudes of input traces
    ncfs /= np.linalg.norm(ncfs, axis=1, keepdims=True)

    b = np.sum(ncfs, axis=0)

    b /= np.max(b)

    return b, None