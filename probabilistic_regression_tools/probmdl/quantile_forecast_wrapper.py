__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array

from probabilistic_regression_tools.scores import crps_for_quantiles

"""
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"""


def iteritems(d, **kw):
    return iter(d.items(**kw))


def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(iteritems(namecount)):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))


class QuantileForecastWrapper(BaseEstimator):
    """  Quantile Forecast Wrapper.

    Class that wraps single quantile forecasts to forecast a cdf of multiple quantiles at once.

    """

    def __init__(self, estimators=None,
                 quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                 loss=crps_for_quantiles,
                 name_of_quantile_param='alpha',
                 include_zero_quantile=False,
                 include_one_quantile=False):
        if estimators is None:
            raise ValueError("Estimators should be a list of quantile forecasts or a single forecast model.")

        self.quantiles = np.array(quantiles)
        self.estimators = estimators
        self.loss = loss
        # self.mapping = collections.defaultdict(type(self.estimator))
        self.include_zero_quantile = include_zero_quantile
        self.include_one_quantile = include_one_quantile

        self.name_of_quantile_param = name_of_quantile_param

        self._init_estimators()

        self.named_est = {key: value for key, value in
                          _name_estimators([self.estimators])}

    def _init_estimators(self):
        if not isinstance(self.estimators, (list,)):
            estimator = self.estimators
            self.estimators = [clone(estimator) for q in self.quantiles]

        if len(self.estimators) != len(self.quantiles):
            raise ValueError("Number of estimators %r must be equal to number of quantiles %r" \
                             % len(self.estimators) % len(self.quantiles))

        self.mapping = dict(zip(self.quantiles, self.estimators))

        for k in self.mapping: self.mapping[k].set_params(alpha=k)

    def fit(self, X, y):
        """
        Note: X and y is not stored, because it takes to much memory
        :param X:
        :param y:
        :return:
        """

        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.estimators = [cur_model.fit(self.X_, self.y_) for cur_model in self.estimators]

        return self

    def predict(self, X):

        # TODO check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        if 0.5 in self.mapping:
            return self.mapping[0.5].predict(X)
        else:
            raise NotImplementedError("Should implement a interpolated forecast " + \
                                      "of the two closest quantiles.")

    def _quantiles_including_zero_and_one_quantile_padding(self):

        quantiles = self.quantiles

        if self.include_zero_quantile: quantiles = [0, *quantiles]
        if self.include_one_quantile: quantiles = [*quantiles, 1]

        return np.array(quantiles)

    def predict_proba(self, X):

        # check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        quantiles = self._quantiles_including_zero_and_one_quantile_padding()

        res = np.zeros((len(quantiles), len(X)))

        for idx, q in enumerate(quantiles):
            if idx == 0 and self.include_zero_quantile:
                cur_res = np.ones(len(X)) * min(self.y_)
            elif idx == len(quantiles) - 1 and self.include_one_quantile:
                cur_res = np.ones(len(X)) * max(self.y_)
            else:
                cur_res = self.mapping[q].predict(X)

            res[idx, :] = cur_res

        return res.T

    def score(self, X, y):
        return crps_for_quantiles(self.predict_proba(X), y, \
                                  self._quantiles_including_zero_and_one_quantile_padding())[0]

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'quantiles':
                self.quantiles = value
            elif key == 'loss':
                self.loss = value
            else:
                self.estimator.set_params(**{key: value})
