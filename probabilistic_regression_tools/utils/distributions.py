""" Conversion from scipy.stats rv_continuous to quantiles. """

import numpy as np

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


    ## TODO should also provide the other way around
