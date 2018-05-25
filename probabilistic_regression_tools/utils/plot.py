import matplotlib.pyplot as plt
import numpy as np


def plot_quantile_forecast(x, measurements, probabilistic_forecasts, quantiles):
    """ TODO

    :param x: The (one dimensional) data for the x-axis.
    :param  measurements: array_like
               List or numpy array with "M" measurements / observations.
    :param  probabilistic_forecasts: array_like  2D-numpy array with quantile forecasts with dimensionality M x Q,
               where "Q" is number of quantiles.
    :param  quantiles: array_like
               List of "Q" values of the quantiles to be evaluated.
    :return:
    """
    x_t = (x * np.ones_like(probabilistic_forecasts))
    plt.scatter(x, measurements, label='M')

    for i in range(x_t.shape[1]):
        plt.scatter(x_t[:, i], probabilistic_forecasts[:, i], label=str(quantiles[i]))

    plt.ylabel("Measurements [M]")
    plt.legend()
    plt.show()
