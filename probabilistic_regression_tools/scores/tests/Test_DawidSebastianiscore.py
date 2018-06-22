"""Test for DSS Score."""

import numpy as np
from nose.tools import assert_true
from scipy.stats import norm

import probabilistic_regression_tools.scores.dawidsebastianiscore as dawidsebastianiscore


class Test_DawidSebastianiscore:
    """Test class for DSS."""

    def test_compare_different_expectations(self):
        """Test that compares same distance between meas and pd."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]
        mean_dss1, single_dss1 = dawidsebastianiscore(meas, pd)

        manualresults = list(map(self._compute_dawid_sebastiani_score, pd, meas))

        is_good = np.equal(single_dss1, manualresults).all()
        assert_true(is_good, msg="Individual DS score values should return same value.")

    def _compute_dawid_sebastiani_score(self, density_func, measurement):
        """Direct computation of dss score."""

        expectation_val = density_func.mean()
        std_dev_val = density_func.std()

        dss1 = np.power(measurement - expectation_val, 2) / np.power(std_dev_val, 2)
        dss2 = 2 * np.log(std_dev_val)

        return dss1 + dss2
