""" CRPS Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import scipy.integrate as integrate


def crps(probabilistic_forecasts, measurements):
    """ Computes the CRPS score with actual integration.

    This variant is the one proposed in Hersbach H. Decomposition of the Continuous Ranked Probability Score for
    Ensemble Prediction Systems. Weather Forecast. 2000;15(5):559-570.

    Parameters
    ----------
        probabilistic_forecasts: list
           List of "M" scipy.stats.rv_continuous distributions
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
        measurements: array_like
           List or numpy array with "M" measurements / observations.

    Returns
    -------
        mean_crps: float
            The mean CRPS over all probabilistic_forecast - measurement pairs.
        single_crps: array, shape (M,)
            CRPS value for each probabilistic_forecast - measurement pair.

    """

    single_crps = np.array(list(map(_computeCRPS, probabilistic_forecasts, measurements)))
    mean_crps = np.mean(single_crps)

    return mean_crps, single_crps


def _computeCRPS(density_func, measurement):
    # integrate from -inf to current measurement
    def func_lower_cdf(x):
        return np.power(density_func.cdf(x), 2)

    part_1 = integrate.quad(func_lower_cdf, -np.inf, measurement)

    # integrate from measurement to inf
    def func_upper_cdf(x):
        return np.power(1 - density_func.cdf(x), 2)

    part_2 = integrate.quad(func_upper_cdf, measurement, np.inf)

    # return sum
    return part_1[0] + part_2[0]
