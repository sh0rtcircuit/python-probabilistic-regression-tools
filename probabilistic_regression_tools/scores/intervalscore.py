""" Intervalscore Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles
import scipy.stats


def intervalscore(probabilistic_forecasts, measurements, quantiles=np.linspace(0.1, 0.9, 9)):
    """ Computes the intervalscore (is).

        Definition of the score is taken from
        Gneiting T, Raftery AE. Strictly Proper Scoring Rules, Prediction, and Estimation.
        J Am Stat Assoc. 2007;102(477):359-378.

        Parameters
        ----------
            probabilistic_forecasts: list
               List of "M" scipy.stats.rv_continuous distributions
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
               OR
               2D-numpy array with quantile forecasts with dimensionality M x Q,
               where "Q" is number of quantiles.
            measurements: array_like
               List or numpy array with "M" measurements / observations.
           quantiles: array_like
               List of "Q" values of the quantiles to be evaluated.

        Returns
        -------
            mean_is: float
                The mean is over all probabilistic_forecast - measurement pairs.
            single_is: array, shape (M,)
                is value for each probabilistic_forecast - measurement pair.
    """

    # convert to quantile representation if necessary
    if isinstance(probabilistic_forecasts[0], scipy.stats._distn_infrastructure.rv_frozen):
        quantile_forecasts = probdists_2_quantiles.probdists_2_quantiles(probabilistic_forecasts, quantiles)
    else:
        quantile_forecasts = np.array(probabilistic_forecasts)

    nr_meas = quantile_forecasts.shape[0]

    nr_quantiles = quantiles.size
    nr_intervals = int(quantiles.size / 2)
    alphas = 2 * quantiles[0:nr_intervals]

    if np.mod(quantiles.size, 2) > 0:
        print(['Number of quantileValues is odd. Dont worry, the median quantile just will not be evaluated.'])

    interval_score_each_obs = np.zeros([nr_meas, nr_intervals])
    for i in range(nr_intervals):
        u_idx = nr_quantiles - i - 1
        l_idx = i

        interval_score_each_obs[:, i] = (quantile_forecasts[:, u_idx] - quantile_forecasts[:, l_idx]) \
                                        + (2 / alphas[i]) * (
            measurements - quantile_forecasts[:, u_idx]) * np.heaviside(
            measurements - quantile_forecasts[:, u_idx], 1) \
                                        + (2 / alphas[i]) * (
            quantile_forecasts[:, l_idx] - measurements) * np.heaviside(
            quantile_forecasts[:, l_idx] - measurements, 1)

    single_is = np.mean(interval_score_each_obs, 0)

    return np.mean(single_is), single_is
