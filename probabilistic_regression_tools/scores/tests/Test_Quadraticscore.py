"""Test for Quadratic Score."""

import numpy as np
import scipy.integrate as integrate
from nose.tools import assert_true
from scipy.stats import norm

import probabilistic_regression_tools.scores.quadraticscore as quadraticscore


class Test_Quadraticscore:
    """Test class for Quadratic."""

    def test_compare_different_expectations(self):
        """Test that compares same distance between meas and pd."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]
        mean_quadratic1, single_quadratic1 = quadraticscore(meas, pd)

        manualresults = list(map(self._compute_quadratic_score, pd, meas))

        is_good = np.equal(single_quadratic1, manualresults).all()
        assert_true(is_good, msg="Individual Quadraticscore values should return same value.")

    def _compute_quadratic_score(self, density_func, measurement):
        """Explicit computation of quadratic score."""
        meas_pdf = density_func.pdf(measurement)

        int_func = lambda x: np.power(density_func.pdf(x), 2)
        second_qs_part = integrate.quad(int_func, -np.inf, np.inf)[0]

        return 2 * meas_pdf - second_qs_part
