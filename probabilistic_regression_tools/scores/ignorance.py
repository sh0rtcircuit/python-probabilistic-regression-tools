''' Ignorance Scoring Rule'''

__author__ = "André Gensler"
__copyright__ = "Copyright 2018"
__status__ = "Prototype"

import numpy as np


def ignorance(measurements, probabilistic_forecasts):
    """ Computes the ignorance score (ign).

        Definition of the score is taken from
        Tödter J, Ahrens B. Generalization of the Ignorance Score: Continuous Ranked Version and Its Decomposition.
        Mon Weather Rev. 2012;140(6):2005-2017.

        Parameters
        ----------
            probabilistic_forecasts: list
               List of "M" scipy.stats.rv_continuous distributions
               https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
            measurements: array_like
               List or numpy array with "M" measurements / observations.

        Returns
        -------
            mean_ign: float
                The mean ign over all probabilistic_forecast - measurement pairs.
            single_ign: array, shape (M,)
                ign value for each probabilistic_forecast - measurement pair.

    """

    density_values = list(map(lambda x, y: x.pdf(y), probabilistic_forecasts, measurements))
    single_ign = - np.log(density_values)

    return np.mean(single_ign), single_ign
