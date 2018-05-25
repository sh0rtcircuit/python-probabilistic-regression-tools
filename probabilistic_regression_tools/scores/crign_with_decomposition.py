""" CRIGN Scoring Rule With Decomposition"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import numpy as np
import scipy.stats

from probabilistic_regression_tools.probdists_2_quantiles import probdists_2_quantiles


def crign_with_decomposition(probabilistic_forecasts, measurements, quantiles=np.linspace(0.1, 0.9, 9)):
    """ Computes the CRIGN score and its decompositions.

        The decomposition is described in
        Tödter J, Ahrens B. Generalization of the Ignorance Score: Continuous Ranked Version and Its Decomposition.
        Mon Weather Rev. 2012;140(6):2005-2017.

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
            crign: float
                The mean CRIGN over all probabilistic_forecast - measurement pairs reconstructed from the decomposition.
            crign_rel: float
                Reliability error of the crign.
            crign_res: float
                Resolution of the crign.
            crign_unc: float
                Uncertainty of the crign.
    """

    # convert to quantile representation if necessary
    if isinstance(probabilistic_forecasts[0], scipy.stats._distn_infrastructure.rv_frozen):
        quantile_forecasts = probdists_2_quantiles(probabilistic_forecasts, quantiles)
    else:
        quantile_forecasts = np.array(probabilistic_forecasts)

    # sort in case of quantile crossing
    quantile_forecasts = np.sort(quantile_forecasts, axis=1)

    # get all possible intervals for weighting
    all_data_pre = np.hstack([quantile_forecasts, np.atleast_2d(measurements).T])
    all_data = np.sort(np.unique(all_data_pre))
    intervals = np.diff(all_data)

    ign_all = np.zeros(all_data.size - 1)
    ign_rel = np.zeros(all_data.size - 1)
    ign_res = np.zeros(all_data.size - 1)
    ign_unc = np.zeros(all_data.size - 1)

    for idx, _ in enumerate(intervals):
        # define binary problem
        threshold = all_data[idx]
        binary_measurement = measurements <= threshold
        quant2 = threshold <= quantile_forecasts
        quant2[:, -1] = 1  # last quantile in any case if no other is hit

        # get highest fitting probability category
        cs = np.cumsum(quant2, axis=1)
        hit_dim1, hit_dim2 = np.where(cs == 1)
        probs = quantiles[hit_dim2].T

        # compute score for binary problem
        [ign_all[idx], ign_rel[idx], ign_res[idx], ign_unc[idx]] = _binaryignorance_with_decomposition(probs,
                                                                                                       binary_measurement)

    # weighted aggregation
    crign = np.sum(intervals * ign_all)
    crign_rel = np.sum(intervals * ign_rel)
    crign_res = np.sum(intervals * ign_res)
    crign_unc = np.sum(intervals * ign_unc)

    return crign, crign_rel, crign_res, crign_unc


def _binaryignorance_with_decomposition(probability_class: object, measurements: object) -> object:
    """Computation of decomposition components of binary ignorance."""
    probability_categories = np.unique(probability_class)
    number_probability_categories = probability_class.size
    mean_measurement = np.mean(measurements)

    # compute decomposition components
    rel, res = np.apply_along_axis(
        lambda x: _binaryignorance_single_decomp(probability_class, measurements, x, number_probability_categories,
                                                 mean_measurement), 0, np.atleast_2d(probability_categories))
    unc = np.array(-mean_measurement * np.log(mean_measurement) - (1 - mean_measurement) * np.log(1 - mean_measurement))
    unc[np.isnan(unc)] = 0

    ign = np.sum(rel) - np.sum(res) + np.sum(unc)

    return ign, np.sum(rel), np.sum(res), unc


def _binaryignorance_single_decomp(probability_class, measurements, current_probability_category,
                                   number_probability_categories, mean_measurement):
    """Decomposition for a single probability level."""
    # compute probability components
    cat_idx = probability_class == current_probability_category

    nr_prob_class = np.sum(cat_idx)
    pyi = nr_prob_class / number_probability_categories
    barzi = np.sum(measurements[cat_idx]) / nr_prob_class
    yi = current_probability_category

    # compute reliability component
    rel1 = pyi * barzi * np.log(barzi / yi)
    rel2 = pyi * (1 - barzi) * np.log((1 - barzi) / (1 - yi))
    rel = np.array([rel1, rel2])
    rel[np.isnan(rel)] = 0

    # compute resolution component
    res1 = pyi * barzi * np.log(barzi / mean_measurement)
    res2 = pyi * (1 - barzi) * np.log((1 - barzi) / (1 - mean_measurement))
    res = np.array([res1, res2])
    res[np.isnan(res)] = 0

    return np.sum(rel), np.sum(res)
