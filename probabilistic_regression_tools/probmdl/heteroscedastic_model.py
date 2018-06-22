import numpy as np
import scipy.stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import probabilistic_regression_tools.scores.crps as crps

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"


class HeteroscedasticRegression(BaseEstimator):
    """  Heteroscedastic regression model.

    Heteroscedastic Forecasting Model based on arbitrary deterministic model
    with Gaussian uncertainty assumption and linear model for uncertainty estimation.

    """

    def __init__(self, deterministic_mdl, loss_func=crps, verbose=1, optimize_method='BFGS'):
        """ Initialize the homoscedastic model with a pretrained deterministic_mdl.

            Parameters
            ----------
                deterministic_mdl: object (sklearn BaseEstimator)
                    A pretrained (deterministic) regression model.
                loss_func: function
                    Function handle to scoring rule which takes arguments
                    loss_func(list of scipy.stats.rv_continuous, array_like measurements).
                verbose: bool
                    Displays optimization progress if True.
                optimize_method: Optimization method from scipy.optimize.
        """

        self.deterministic_model = deterministic_mdl
        self.loss_func = loss_func
        self._linearmodel_coeffs = np.array([0, 1])
        self.epsilon = np.finfo(float).eps
        self.verbose = verbose
        self.it = 0
        self.optimize_method = optimize_method

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
        opt_result = minimize(lambda linear_coeffs: self._fitness(X, y, linear_coeffs), self._linearmodel_coeffs,
                              method=self.optimize_method)
        self._linearmodel_coeffs = opt_result.x
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
        det_pred = self.deterministic_model.predict(X)
        sigma_pred = self._compute_sigma(det_pred)
        return list(map(scipy.stats.norm, det_pred.tolist(), sigma_pred.tolist()))

    def _fitness(self, X, y, _linearmodel_coeffs):
        """Fitness function."""
        self._linearmodel_coeffs = _linearmodel_coeffs
        prob_pred = self.predict(X)
        lossfunc_val = self.loss_func(y, prob_pred)[0]

        if self.verbose>0:
            print('Iteration ' + str(self.it) + ' loss func value: ' + str(lossfunc_val))
            self.it = self.it + 1

        return lossfunc_val

    def _compute_sigma(self, expectation_value):
        """Computes the sigma value depending on the expectation value."""
        return np.maximum(self._linearmodel_coeffs[1] * expectation_value + self._linearmodel_coeffs[0], self.epsilon)
