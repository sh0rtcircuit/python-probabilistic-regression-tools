from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array

from probabilistic_regression_tools.scores import crps_for_quantiles

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

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
                 include_one_quantile=False,
                 predict_cdfs=False):
        """
        :param estimators:
                A single estimator to fit or a list of different estimators.
                In case a single estimator is given the estimator is cloned.
                 In case a list of estimators is given they need to aligned with the quantiles.
        :param quantiles: list or numpy.array
                One dimenstional array or list of quantiles that are fitted by the estimators.
        :param loss: probabilistic loss
                Loss that is used in the score function, e.g., for GridSearch.
        :param name_of_quantile_param: str
                The name of parameter to set the quantile for each estimator.
        :param include_zero_quantile: bool
                True if the minimum of the training data should be predicted as 0th quantile.
        :param include_one_quantile: bool
              True if the maximum of the training data should be predicted as 1th quantile.
        :param predict_cdfs: bool
                False if quantiles should be forecasted in the predict_proba method.
                True if the output should be converted to CDFs.
        """

        if estimators is None:
            raise ValueError("Estimators should be a list of quantile forecasts or a single forecast model.")

        # in case only a single quantile is given
        if not isinstance(quantiles, (list,)):
            quantiles = [quantiles]

        self.quantiles = np.array(quantiles)
        self.estimators = estimators
        self.loss = loss
        self.include_zero_quantile = include_zero_quantile
        self.include_one_quantile = include_one_quantile

        self.name_of_quantile_param = name_of_quantile_param
        self.mapping = dict()

        self._init_estimators()

        self.predict_cdfs = predict_cdfs

        self.named_est = {key: value for key, value in
                          _name_estimators([self.estimators])}

    def _init_estimators(self):
        """Sets the different quantiles for all estimators"""
        if not isinstance(self.estimators, (list,)):
            estimator = self.estimators
            self.estimators = [clone(estimator) for q in self.quantiles]

        if len(self.estimators) != len(self.quantiles):
            raise ValueError("Number of estimators %r must be equal to number of quantiles %r" \
                             % len(self.estimators) % len(self.quantiles))

        self.mapping = dict(zip(self.quantiles, self.estimators))

        args = dict()
        for k in self.mapping:
            args[self.name_of_quantile_param] = k
            self.mapping[k].set_params(**args)

    def fit(self, X, y):
        """ Fit the model to create distributions with correct spread.

            Parameters
            ----------
                X : array_like, shape (n, n_features)
                    List of n_features-dimensional data points. Each row
                    corresponds to a single data point.
                y : array_like, shape (n,)
                    Labels for the given data points X
        """

        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        self.estimators = [cur_model.fit(self.X_, self.y_) for cur_model in self.estimators]

        return self

    def predict(self, X):
        """ Creates a predictive distribution for each data point.

            Parameters
            ----------
                X : array-like, shape = [n_samples, n_features]

            Returns
            -------
                probabilistic_forecasts : array, shape = (n_samples,) component memberships
        """

        return self.predict_proba(X)

    def _quantiles_including_zero_and_one_quantile_padding(self):
        """Helper function to get a list of quantiles with 0th and 1th quantile for prediction."""
        quantiles = self.quantiles

        if self.include_zero_quantile: quantiles = [0, *quantiles]
        if self.include_one_quantile: quantiles = [*quantiles, 1]

        return np.array(quantiles)

    def predict_proba(self, X):
        """ Creates a predictive distribution for each data point.

            Parameters
            ----------
                X : array-like, shape = [n_samples, n_features]

            Returns
            -------
                probabilistic_forecasts : array, shape = (n_samples,) component memberships
        """

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

        if self.predict_cdfs:
            raise NotImplementedError("Should implement a conversion to CDFS.")

        return res.T

    def score(self, X, y):
        """ Return the configured probabilistic score.

            Parameters
            ----------
                X : array_like, shape (n, n_features)
                    List of n_features-dimensional data points. Each row
                    corresponds to a single data point.
                y : array_like, shape (n,)
                    Labels for the given data points X
        """

        return crps_for_quantiles(self.predict_proba(X), y, \
                                  self._quantiles_including_zero_and_one_quantile_padding())[0]

    def set_params(self, **params):
        """Set the parameters of this estimator.
                The method works on simple estimators as well as on nested objects
                (such as pipelines). The latter have parameters of the form
                ``<component>__<parameter>`` so that it's possible to update each
                component of a nested object.
                Returns
                -------
                self
        """

        for key, value in params.items():
            if key == 'quantiles':
                self.quantiles = value
            elif key == 'loss':
                self.loss = value
            else:
                self.estimator.set_params(**{key: value})
        return self