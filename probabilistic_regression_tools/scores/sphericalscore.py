""" Sphericalscore Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import scipy.integrate as integrate


def sphericalscore(probabilistic_forecasts, measurements):
    """ Computes the spherical score (sphs).

    Definition of the score is taken from
    Gneiting T, Raftery AE. Strictly Proper Scolring Rules, Prediction, and Estimation.
    J Am Stat Assoc. 2007;102(477):359-378.

    Parameters
    ----------
        probabilistic_forecasts: list
           List of "M" scipy.stats.rv_continuous distributions
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
        measurements: array_like
           List or numpy array with "M" measurements / observations.

    Returns
    -------
        mean_sphs: float
            The mean sphs over all probabilistic_forecast - measurement pairs.
        single_sphs: array, shape (M,)
            sphs value for each probabilistic_forecast - measurement pair.
    """

    single_sphs = np.array(list(map(_compute_spherical_score, probabilistic_forecasts, measurements)))

    return np.mean(single_sphs), single_sphs


def _compute_spherical_score(density_func, measurement):
    """Computation of spherical score of a single density_func - measurement pair."""
    meas_pdf = density_func.pdf(measurement)

    def int_func(x):
        return np.power(density_func.pdf(x), 2)

    second_sphs_part = np.sqrt(integrate.quad(int_func, -np.inf, np.inf)[0])

    return meas_pdf / second_sphs_part
