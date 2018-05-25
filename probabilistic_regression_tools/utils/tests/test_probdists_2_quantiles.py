# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
from nose.tools import assert_true

from probabilistic_regression_tools.utils import probdists_2_quantiles


class Test_Probdists_2_Quantiles:
    """Test class for CRPS."""

    def test_probdists_2_quantiles(self):
        normdist1 = scipy.stats.norm(0, 1)
        normdist2 = scipy.stats.norm(1, 2)

        normdist_list = [normdist1, normdist2]
        quantiles = np.linspace(0.1, 0.9, 9)

        correct_answer = np.array([normdist1.ppf(quantiles), normdist2.ppf(quantiles)])
        result = probdists_2_quantiles(normdist_list, quantiles)

        is_good = np.equal(correct_answer, result).all()
        assert_true(is_good, msg="Unexpected difference in ppf function.")
