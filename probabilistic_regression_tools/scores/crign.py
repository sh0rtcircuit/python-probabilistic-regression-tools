""" CRIGN Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import scipy.integrate as integrate


def crign(measurements, probabilistic_forecasts):
    """ Computes the CRIGN score with actual integration.

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
        mean_crign: float
            The mean CRIGN over all probabilistic_forecast - measurement pairs.
        single_crign: array, shape (M,)
            CRIGN value for each probabilistic_forecast - measurement pair.

    """

    if len(probabilistic_forecasts) != len(measurements):
        raise ValueError('Length of lists of first two arguments have to be equal.')

    single_crign = list(map(_compute_crign, probabilistic_forecasts, measurements))

    mean_crign = np.mean(np.array(single_crign))

    return mean_crign, single_crign


def _compute_crign(density_func, measurement):
    """Computes the CRIGN for a single density function - measurement pair."""

    # integrate from -inf to current measurement
    def func_lower_cdf(x):
        return np.log(np.abs(density_func.cdf(x) - 1))

    part_1 = integrate.quad(func_lower_cdf, -np.inf, measurement)

    # integrate from measurement to inf
    def func_upper_cdf(x):
        return np.log(np.abs(density_func.cdf(x) - 0))

    part_2 = integrate.quad(func_upper_cdf, measurement, np.inf)

    # return sum
    return -(part_1[0] + part_2[0])
