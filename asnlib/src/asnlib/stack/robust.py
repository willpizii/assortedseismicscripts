import numpy as np

# As described by Pavlis and Vernon, 2010
# Shown by Yang et al. 2023 to be good for phase dispersion and other analysis

def robust_stack(ncfs, eps = 1e-5, max_iters = 2000):
    """
    Takes given signals, such as ncfs, and returns the robust stack
    Defined by Pavlis and Vernon, 2010
    
    Parameters:
    ncfs: np.stack of traces to process; in stack of N traces x M samples
    eps: convergence criterion epsilon
    max_iters: maximum number of iterations (only useful for non-converging stacks)

    Returns:
    b: final weighted robust stack
    weights: weights of each input data trace
    """

    # cast input trace stack to array
    ncfs = np.asarray(ncfs, dtype=float)

    # normalise amplitudes of input traces
    ncfs /= np.linalg.norm(ncfs, axis=1, keepdims=True)

    # take shape - N number of traces, M trace length
    N, M = ncfs.shape

    # raw median as initial reference trace
    b = np.median(ncfs, axis=0)

    # working loop
    for _ in range(max_iters):

        # initialise weights as 0-array
        weights = np.zeros(N)

        # iterate through traces to stack
        for i, d in enumerate(ncfs):

            # latest reference stack as bj, and sample trace d as di
            bj = b
            di = d

            # define weight update function
            top = np.linalg.norm(np.dot(bj,di))

            ri = bj - np.dot(bj,di) * di
            bottom = np.linalg.norm(di) * np.linalg.norm(ri)
            
            # update weight for given trace
            weights[i] = top / bottom
        
        # normalise
        weights /= np.sum(weights)

        # update reference stack
        b_u = np.sum(weights[:, None] * ncfs, axis=0)

        # check against last stack for convergence
        crit = np.linalg.norm(b_u - b, 1) / np.linalg.norm(b_u, 2) * M

        # update reference stack
        b=b_u

        # break if convergence criterion met
        if crit < eps:
            break
        
    return b, weights