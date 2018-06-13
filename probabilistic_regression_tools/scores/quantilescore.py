""" Quantile Score Scoring Rule"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import numpy as np
#import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles
from probabilistic_regression_tools.utils import probdists_2_quantiles
import scipy.stats


def quantilescore(probabilistic_forecasts, measurements, quantiles=np.linspace(0.1, 0.9, 9)):
    """ Computes the quantile score (qs).

        Definition of the score is taken from
        Bentzien S, Friederichs P. Decomposition and graphical portrayal of the quantile score.
        Q J R Meteorol Soc. 2014;140(683):1924-1934.

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
            mean_qs: array, shape (Q,)
                The mean qs over all probabilistic_forecast - measurement pairs for each quantile.
            single_qs: array, shape (M,Q)
                qs value for each probabilistic_forecast - measurement pair for each quantile.
    """

    if isinstance(probabilistic_forecasts[0], scipy.stats._distn_infrastructure.rv_frozen):
        quantile_forecasts = probdists_2_quantiles(probabilistic_forecasts, quantiles=quantiles)
    else:
        quantile_forecasts = np.array(probabilistic_forecasts)

    quantiles = np.atleast_1d(quantiles)
    meas_rep = np.tile(np.array(measurements).reshape(len(measurements), 1), [1, quantiles.size])

    distances = meas_rep - quantile_forecasts
    single_qs = _pinball_loss(distances, quantiles)

    return np.mean(single_qs, axis=0), single_qs


def _pinball_loss(distances, tau):
    """Computes the actual quantile score loss function."""

    # repeat quantile
    if tau.size == 1:
        tau = np.tile(tau, distances.shape)
    else:
        tau = np.tile(tau, [distances.shape[0], 1])

    # for smaller 0 and bigger 0 at the same time
    return np.abs(np.maximum(0, distances)) * tau + np.abs(np.minimum(0, distances)) * (1 - tau)
