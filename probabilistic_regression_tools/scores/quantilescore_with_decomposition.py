""" Quantile Score Scoring Rule with decomposition."""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import numpy as np
import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles
import scipy.stats


def quantilescore_with_decomposition(probabilistic_forecasts, measurements, quantiles=np.linspace(0.1, 0.9, 9), K=10,
                                     bin_averaging=False):
    """ Computes the quantile score (qs) and its decompositions.

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
            K: int
                Subsampling parameter of qs decomposition. Defines the amount of discretization steps.
            bin_averaging: bool
                Defines whether the discretized quantile forecasts are defined in the center
                or the edge of the subsampling bin. According to the authors, no averaging is more correct
                while averaging may lead to more stable results.
        Returns
        -------
            qs: array, shape (Q,)
                The mean qs over all probabilistic_forecast - measurement pairs for each quantile recomposed
                from the decomposition components. Not (!) completely identical to the non-decomposed quantilescore.
            qs_rel: array, shape (Q,)
                Reliability error of the qs for each quantile.
            qs_res: array, shape (Q,)
                Resolution of the qs for each quantile.
            qs_unc: array, shape (Q,)
                Uncertainty of the qs for each quantile.
    """

    if isinstance(probabilistic_forecasts[0], scipy.stats._distn_infrastructure.rv_frozen):
        quantile_forecasts = probdists_2_quantiles.probdists_2_quantiles(probabilistic_forecasts, quantiles=quantiles)
    else:
        quantile_forecasts = np.array(probabilistic_forecasts)

    quantiles = np.atleast_1d(quantiles)

    # DECOMPOSITION COMPUTATION OF SCORE
    nr_q = quantiles.size
    smeasurements = np.sort(measurements)

    qs = [[] for i in range(nr_q)]
    qs_rel = [[] for i in range(nr_q)]
    qs_res = [[] for i in range(nr_q)]
    qs_unc = [[] for i in range(nr_q)]

    for i in range(0, nr_q):

        # compute the unconditional sample quantile mean_meas_tau ==> \bar{o}_tau
        # eventually average value instead of quantile upper bound ?
        qIdx = int(quantiles[i] * measurements.size) - 1
        if bin_averaging:
            # average value
            mean_meas_tau = np.mean(smeasurements[0:qIdx])
        else:
            # upper bound
            mean_meas_tau = smeasurements[qIdx]
        # uncQ = np.mean(_pinball_loss(measurements - mean_meas_tau, quantiles[i]))

        # prepare indices(for binning with k) and ranges(for weighting at the end)
        cur_qe = quantile_forecasts[:, i]
        s_idx = np.argsort(cur_qe)
        idx_borders = np.linspace(0, s_idx.size, K + 1)
        ranges = np.diff(idx_borders)

        # iterate on subsets with k (the conditional quantiles)
        entropy_k = [[] for someIdx in range(K)]
        unc_k = np.zeros([K, 1])
        res_k = np.zeros([K, 1])
        rel_k = np.zeros([K, 1])

        for k in range(K):
            # get borders of current subset
            k_start = int(idx_borders[k])
            k_end = int(idx_borders[k + 1]) - 1

            # specify subsets from borders
            idx_range = s_idx[k_start:(k_end + 1)]
            cur_qe_k = cur_qe[idx_range]
            measurements_k = measurements[idx_range]

            # create conditional measurements o_tau^(k) and p_tau^(k)
            q_idxk = int(quantiles[i] * measurements_k.size)
            smeasurements_k = np.sort(measurements_k)

            if bin_averaging:
                # average value
                mean_measurement_k = np.mean(smeasurements_k[0:q_idxk])
            else:
                # the conditional sample quantile observation o_tau^(k) upper bound
                mean_measurement_k = smeasurements_k[q_idxk]

            # the conditional discretized quantile forecast p_tau^(k)
            fcst_k = np.mean(cur_qe_k)

            # compute decomposition components
            entropy_k[k] = _pinball_loss(measurements_k - mean_measurement_k, quantiles[i])
            rel_k[k] = np.mean(_pinball_loss(measurements_k - fcst_k, quantiles[i]) - entropy_k[k])
            res_k[k] = np.mean(_pinball_loss(measurements_k - mean_meas_tau, quantiles[i]) - entropy_k[k])
            unc_k[k] = np.mean(_pinball_loss(measurements_k - mean_meas_tau, quantiles[i]))

        # compute the weights for overall score
        # this is N_k / N
        relative_weights = ranges / np.sum(ranges)

        # create summary values
        qs_rel[i] = np.sum(relative_weights * rel_k.ravel())
        qs_res[i] = np.sum(relative_weights * res_k.ravel())
        qs_unc[i] = np.sum(relative_weights * unc_k.ravel())
        qs[i] = qs_rel[i] - qs_res[i] + qs_unc[i]

    return np.array(qs), np.array(qs_rel), np.array(qs_res), np.array(qs_unc)


def _pinball_loss(distances, tau):
    """Computes the actual quantile score loss function."""

    # repeat quantile
    if tau.size == 1:
        tau = np.tile(tau, distances.shape)
    else:
        tau = np.tile(tau, [distances.shape[0], 1])

    # for smaller 0 and bigger 0 at the same time
    return np.abs(np.maximum(0, distances)) * tau + np.abs(np.minimum(0, distances)) * (1 - tau)
