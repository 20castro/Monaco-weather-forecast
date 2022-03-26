import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from typing import Tuple

def normal(x, m, s):
    return np.exp(-(x - m)**2/(2*s*s))/(np.sqrt(2*np.pi)*s)

def approx(ax: plt.Axes, err: np.ndarray, hist_name: str, color: Tuple[str], plot_fit: bool):

    val, key, _ = ax.hist(
        err,
        bins=100,
        alpha=.5,
        density=True,
        label=hist_name + ' error distribution',
        color=color[0])
        
    if plot_fit:
        try:
            key = .5*(key[1:] + key[:-1])
            opt, _ = curve_fit(normal, key, val, bounds=(np.array([-np.inf, 0]), np.array([np.inf, np.inf])))
            m, s = opt[0], opt[1]
            ax.plot([m, m], [0, 1./(np.sqrt(2.*np.pi)*s)], ':', color=color[1])
            ax.plot(key, normal(key, m, s), '--', color=color[1], label=hist_name + ' gaussian fit')
        except OptimizeWarning:
            pass

    plt.legend()