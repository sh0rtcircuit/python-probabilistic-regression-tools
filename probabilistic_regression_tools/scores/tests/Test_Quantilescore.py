"""Test for Quantile Score"""

import numpy as np
from nose.tools import assert_true
from scipy.stats import norm

from probabilistic_regression_tools.scores.quantilescore import quantilescore


class Test_Quantilescore:
    """Test Class for quantile score."""

    def test_with_quantiles(self):
        """Tests the functionality with quantiles as input."""

        quantile_levels = np.linspace(0.1, 0.9, 9)
        quantile_forecast = np.linspace(-0.4, 0.4, 9)

        quantile_forecasts = [quantile_forecast, quantile_forecast, quantile_forecast]

        meas = np.array([-0.3, 0, 0.1])

        mean_qs, single_qs = quantilescore(quantile_forecasts, meas, quantiles=quantile_levels)

        # reference implementation by hand
        qf2 = np.array(quantile_forecasts)
        meas_rep = np.tile(meas.reshape(3, 1), [1, qf2.shape[1]])
        errors = Test_Quantilescore._pinball_loss(meas_rep - qf2, quantile_levels)

        is_good = np.equal(single_qs, errors).all()
        assert_true(is_good, msg="Results should be equal.")

    def test_with_distribution(self):
        """Test with probability distribution as input."""

        quantiles = np.linspace(0.1, 0.9, 9)

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]

        mean_qs, single_qs = quantilescore(pd, meas)

        # reference implementation by hand
        qf2 = list(map(lambda x: x.ppf(quantiles), pd))
        qf2 = np.array(qf2)
        meas_rep = np.tile(np.array(meas).reshape(3, 1), [1, qf2.shape[1]])
        errors = Test_Quantilescore._pinball_loss(meas_rep - qf2, quantiles)

        is_good = np.equal(single_qs, errors).all()
        assert_true(is_good, msg="Results should be equal.")

    def test_with_single_tau(self):
        """Test with single tau as input."""
        quantiles = 0.1
        quantile_forecast = np.linspace(-0.4, 0.4, 9)

        quantile_forecasts = [quantile_forecast, quantile_forecast, quantile_forecast]

        meas = np.array([-0.3, 0, 0.1])

        mean_qs, single_qs = quantilescore(quantile_forecasts, meas, quantiles=quantiles)

        # reference implementation by hand
        qf2 = np.array(quantile_forecasts)
        meas_rep = np.tile(meas.reshape(3, 1), [1, qf2.shape[1]])
        errors = Test_Quantilescore._pinball_loss(meas_rep - qf2, np.atleast_1d(quantiles))

        is_good = np.equal(single_qs, errors).all()
        assert_true(is_good, msg="Results should be equal.")

    @staticmethod
    def _pinball_loss(distances, tau):
        """Computes the actual quantile score loss function."""
        # preallocate matrix
        error = np.zeros(distances.shape)

        # repeat quantile
        if tau.size == 1:
            tau = np.tile(tau, error.shape)
        else:
            tau = np.tile(tau, [error.shape[0], 1])

        # smaller 0
        error[distances <= 0] = (tau[distances <= 0] - 1) * distances[distances <= 0]
        # bigger 0
        error[distances > 0] = tau[distances > 0] * distances[distances > 0]

        return error
