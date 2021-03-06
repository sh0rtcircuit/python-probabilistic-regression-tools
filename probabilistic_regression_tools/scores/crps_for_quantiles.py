""" CRPS Scoring Rule For Quantiles"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

# 27.02.2018

import numpy as np
import scipy.stats

import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles


def crps_for_quantiles(probabilistic_forecasts, measurements, quantiles=np.linspace(0.1, 0.9, 9)):
    """ Computes the CRPS score with quantile representation.

        This variant is the variant proposed in Hersbach H. Decomposition of the Continuous Ranked Probability Score for
        Ensemble Prediction Systems. Weather Forecast. 2000;15(5):559-570.

        Parameters
        ----------
            probabilistic_forecasts: array_like
               Either list of "M" scipy.stats.rv_continuous distributions
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
            mean_crps: float
                The mean CRIGN over all probabilistic_forecast - measurement pairs.
            single_crps: array, shape (M,)
                CRIGN value for each probabilistic_forecast - measurement pair.
    """

    # convert to quantile representation if necessary
    if isinstance(probabilistic_forecasts[0], scipy.stats._distn_infrastructure.rv_frozen):
        quantile_forecasts = probdists_2_quantiles.probdists_2_quantiles(probabilistic_forecasts, quantiles)
    else:
        quantile_forecasts = np.array(probabilistic_forecasts)

    measurements = np.atleast_2d(measurements).T

    alpha_mat = np.diff(np.hstack([quantile_forecasts, measurements]))
    alpha_mat = np.maximum(0, alpha_mat)
    alpha_mat = np.minimum(alpha_mat, np.maximum(0, np.repeat(measurements, quantile_forecasts.shape[1],
                                                              axis=1) - quantile_forecasts))

    beta_mat = np.diff(np.hstack([measurements, quantile_forecasts]))
    beta_mat = np.maximum(0, beta_mat)
    beta_mat = np.minimum(beta_mat,
                          np.maximum(0,
                                     quantile_forecasts - np.repeat(measurements, quantile_forecasts.shape[1], axis=1)))

    single_crps = np.matmul(alpha_mat, np.power(quantiles, 2)) + np.matmul(beta_mat, np.power(quantiles - 1, 2))

    return np.mean(single_crps), single_crps
