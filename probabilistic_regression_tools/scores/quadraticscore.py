""" Quadraticscore Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import scipy.integrate as integrate


def quadraticscore(probabilistic_forecasts, measurements):
    """ Computes the quadratic score (quads).

        Definition of the score is taken from
        Gneiting T, Raftery AE. Strictly Proper Scoring Rules, Prediction, and Estimation.
        J Am Stat Assoc. 2007;102(477):359-378.

        Parameters
        ----------
            probabilistic_forecasts: list
               List of "M" scipy.stats.rv_continuous distributions
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
            measurements: array_like
               List or numpy array with "M" measurements / observations.
           quantiles: array_like
               List of "Q" values of the quantiles to be evaluated.

        Returns
        -------
            mean_quads: float
                The mean quads over all probabilistic_forecast - measurement pairs.
            single_quads: array, shape (M,)
                quads value for each probabilistic_forecast - measurement pair.
    """

    single_quads = np.array(list(map(_compute_quadratic_score, probabilistic_forecasts, measurements)))

    return np.mean(single_quads), single_quads


def _compute_quadratic_score(density_func, measurement):
    meas_pdf = density_func.pdf(measurement)

    def int_func(x):
        return np.power(density_func.pdf(x), 2)

    second_quads_part = integrate.quad(int_func, -np.inf, np.inf)[0]

    return 2 * meas_pdf - second_quads_part
