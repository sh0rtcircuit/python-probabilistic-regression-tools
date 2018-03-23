"""Homoscedastic Model"""

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import numpy as np
import scipy.stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


def _log_likelihood_loss(prob_pred, y):
    # compute loglikelihood
    likelihoods = np.array(list(map(lambda x, measurement: x.pdf(measurement), prob_pred, y)))
    return [(1 / y.size) * np.sum(-np.log(likelihoods[likelihoods >= 0]))]


class Homoscedastic_Mdl(BaseEstimator):
    """ Simple Homoscedastic Forecasting Model based on arbitrary deterministic model
    with Gaussian uncertainty assumption and constant spread.
    """

    def __init__(self, deterministic_mdl, loss_func=_log_likelihood_loss, sigma=1, verbose=True):
        """ Initialize the homoscedastic model with a pretrained deterministic_mdl.

            Parameters
            ----------
                deterministic_mdl: object (sklearn BaseEstimator)
                    A pretrained (deterministic) regression model.
                loss_func: function
                    Function handle to scoring rule which takes arguments
                    loss_func(list of scipy.stats.rv_continuous, array_like measurements).
                sigma: float
                    A spread parameter for the Gaussian distribution assumption.
                verbose: bool
                    Displays optimization progress if True.
        """
        self.detMdl = deterministic_mdl
        self.loss_func = loss_func
        self.sigma = sigma
        self.epsilon = np.finfo(float).eps
        self.verbose = verbose
        self.it = 0

    def set_sigma(self, sigma):
        """ Function to directly set sigma parameter.

            Parameters
            ----------
                sigma: float
                    Sets the spread parameter for the Gaussian distribution assumption.
        """
        self.sigma = sigma

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
        opt_result = minimize(lambda sigma: self._fitness(X, y, sigma), 1, method='BFGS')
        self.sigma = opt_result.x

    def predict(self, X):
        """Creates a predictive distribution for each data point.

            Parameters
            ----------
                X : array-like, shape = [n_samples, n_features]

            Returns
            -------
                probabilistic_forecasts : array, shape = (n_samples,) component memberships
        """
        det_pred = self.detMdl.predict(X)
        return list(map(scipy.stats.norm, det_pred.tolist(), self.sigma.tolist() * len(det_pred)))

    def _fitness(self, X, y, sigma):
        """Fitness function."""
        self.sigma = np.maximum(sigma, 0)
        prob_pred = self.predict(X)
        lossfunc_val = self.loss_func(prob_pred, y)[0]

        if self.verbose:
            print('Iteration ' + str(self.it) + ' loss func value: ' + str(lossfunc_val))
            self.it = self.it + 1

        return lossfunc_val
