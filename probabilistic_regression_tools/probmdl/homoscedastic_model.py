import numpy as np
import scipy.stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
import warnings

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"


def _log_likelihood_loss(prob_pred, y):
    # compute loglikelihood
    likelihoods = np.array(list(map(lambda x, measurement: x.pdf(measurement), prob_pred, y)))
    warnings.filterwarnings("ignore")
    # TODO: throws divide by zero encountered in double_scalars sometimes 
    return [(1 / y.size) * np.sum(-np.log(likelihoods[likelihoods >= 0]))]


class HomoscedasticRegression(BaseEstimator):
    """ Homoscedastic regression model.

        Homoscedastic Forecasting Model based on arbitrary deterministic model
        with Gaussian uncertainty assumption and constant spread.
    """

    def __init__(self, deterministic_mdl, loss_func=_log_likelihood_loss, sigma=1, \
                 verbose=1, optimize_method='BFGS'):
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
                optimize_method: Optimization method from scipy.optimize.

        """
        self.detMdl = deterministic_mdl
        self.loss_func = loss_func
        self.sigma = sigma
        self.epsilon = np.finfo(float).eps
        self.verbose = verbose
        self.it = 0
        self.optimize_method = optimize_method

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
        opt_result = minimize(lambda sigma: self._fitness(X, y, sigma), 1, method=self.optimize_method)
        self.sigma = opt_result.x
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
        det_pred = self.detMdl.predict(X)
        return list(map(scipy.stats.norm, det_pred.tolist(), self.sigma.tolist() * len(det_pred)))

    def _fitness(self, X, y, sigma):
        """ Fitness function. """
        self.sigma = np.maximum(sigma, 0)
        prob_pred = self.predict(X)
        lossfunc_val = self.loss_func(prob_pred, y)[0]

        if self.verbose > 0:
            print('Iteration ' + str(self.it) + ' loss func value: ' + str(lossfunc_val))
            self.it = self.it + 1

        return lossfunc_val
