""" Dawid-Sebastiani Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np


def dawidsebastianiscore(probabilistic_forecasts, measurements):
    """ Computes the Dawid Sebastiani (dss) scoring rule.

    Definition of the dss is taken from
    Gneiting T, Katzfuss M. Probabilistic Forecasting. Annu Rev Stat Its Appl. 2014;1(1):125-151.

    Parameters
    ----------
        probabilistic_forecasts: list
           List of "M" scipy.stats.rv_continuous distributions
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
        measurements: array_like
           List or numpy array with "M" measurements / observations.

    Returns
    -------
        mean_dss: float
            The mean dss over all probabilistic_forecast - measurement pairs.
        single_dss: array, shape (M,)
            dss value for each probabilistic_forecast - measurement pair.
    """

    single_dss = np.array(list(map(_compute_dawid_sebastiani_score, probabilistic_forecasts, measurements)))

    return np.mean(single_dss), single_dss


def _compute_dawid_sebastiani_score(density_func, measurement):
    expectation_val = density_func.mean()
    std_dev_val = density_func.std()

    dss_1 = np.power(measurement - expectation_val, 2) / np.power(std_dev_val, 2)
    dss_2 = 2 * np.log(std_dev_val)

    return dss_1 + dss_2
