import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

def inverse_power_curve(x, a, b, c):
    return (1-a) - b * np.power(x, c)

def plot_inverse_power_curve(x, y, derivative_threshold=0.00001):
    popt, _ = curve_fit(inverse_power_curve, x, y, bounds=([-np.inf, -np.inf, -1],[np.inf, np.inf, 0]), maxfev=5000)
    last_x = x[-2]

    derivative = 1
    while derivative_threshold <= derivative:
        if inverse_power_curve(last_x, *popt) > 1:
            last_x /= 2
            break
        # Double the last sample size
        last_x *= 2
        # Compute the derivative of the curve at the last point
        derivative = (inverse_power_curve(last_x, *popt) - inverse_power_curve(last_x / 2,*popt)) / (
                                 last_x - last_x / 2)

    sample_sizes_extended = np.linspace(x[0], last_x, 100)
    curve_extended = inverse_power_curve(sample_sizes_extended, *popt)
    plt.plot(sample_sizes_extended, curve_extended, '--', color='gray')


def plot_cv_sample_curve(sample_sizes, cv_scores, y_label='ROC AUC', save_dir=None):
    """
    :param sample_sizes: list of numbers
    :param cv_scores: expects in list of tuples [(a,b,c,d),(e,f,g,h)...]
    :return:
    """
    assert len(cv_scores) == len(sample_sizes)
    mean = np.nanmean(cv_scores, axis=1)
    plt.scatter(sample_sizes, mean, color='black', s=6)

    for i, _ in enumerate(sample_sizes):
        plt.scatter([sample_sizes[i]]*len(cv_scores[i]), cv_scores[i], color='red', s=1)

    plot_inverse_power_curve(sample_sizes, mean)

    plt.xlabel('Sample Size')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
    plt.show()
sample_sizes=[16,32,64,118]
confidence_intervals=[(0.5,0.6,0.7,.6,.5), (0.6, 0.7,.6,.7,.6), (0.65, 0.75,.6,.7,.7), (0.67, 0.8,.8,.7,.7)]
plot_cv_sample_curve(sample_sizes, confidence_intervals)