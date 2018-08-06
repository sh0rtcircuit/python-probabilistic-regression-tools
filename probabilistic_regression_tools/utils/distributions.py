""" Conversion from scipy.stats rv_continuous to quantiles. """

import numpy as np
from scipy import interpolate
from scipy.stats import rv_continuous

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"


def probdists_2_quantiles(probabilistic_forecasts, quantiles=np.linspace(0.1, 0.9, 9)):
    """Converts list of 'M' scipy.stats objects (rv_continuous) to quantile
    representation. 
    
    Parameters
    ----------
        probabilistic_forecasts: list
            List of "M" scipy.stats.rv_continuous distributions
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
        quantiles: array_like
            "Q" values of the quantiles to be evaluated,
            e.g., numpy array with 'Q' elements in range [0..1].

    Returns
    -------
    quantile_forecasts: array, shape(M,Q)
        The resulting quantile forecats.
    """

    quantile_forecasts = list(map(lambda x: x.ppf(quantiles), probabilistic_forecasts))
    return np.array(quantile_forecasts)


class Quantile_2_CDF(rv_continuous):
    def stats(self):
        return 0., 0.

    def _init(self, x_values, quantiles):
        self.x_values = x_values
        self.quantiles = quantiles

    def _cdf(self, x):
        """USE self.quantiles and x_values to return correct values. Ideally via linear interpolation"""
        return np.interp(x, self.x_values, self.quantiles)


def quantiles_2_probdists(quantile_forecasts, quantiles):
    """
        Converts list of 'M' quantile forecasts to scipy.stats objects (rv_continuous)
        representation.

        Parameters
        ----------
            quantile_forecasts: array, shape(M,Q)
                The quantile forecats.
              quantiles: array_like
                "Q" values of the quantiles to be evaluated,
                e.g., numpy array with 'Q' elements in range [0..1].
        Returns
        -------
            probabilistic_forecasts: list
                The resulting List of "M" scipy.stats.rv_continuous distributions
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
    """
    N = np.array(quantile_forecasts).shape[0]
    result = []

    for i in range(N):
        cur_cdf = Quantile_2_CDF(name='mycdf')
        cur_cdf.x_values = quantile_forecasts[i, :]
        cur_cdf.quantiles = quantiles
        result.append(cur_cdf)

    return np.array(result)

