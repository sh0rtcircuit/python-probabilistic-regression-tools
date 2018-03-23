"""Test for ignorance scoring rule."""

import numpy as np
from nose.tools import assert_true
from scipy.stats import norm

import probabilistic_regression_tools.scores.ignorance as ignorance


class Test_Ignorance:
    """Test class for ignorance score."""

    def test_ignorance(self):
        """Compares ignorance score computation with explicit computation."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]

        [meanign, ign] = ignorance.ignorance(pd, meas)

        # for comparison
        ignorance_score = - np.log(list(map(lambda x, y: x.pdf(y), pd, meas)))
        assert_true(np.equal(ign, ignorance_score).all(), msg="Ignorance score not as expected.")
