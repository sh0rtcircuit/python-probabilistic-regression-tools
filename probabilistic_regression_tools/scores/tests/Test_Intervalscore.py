"""Test for Interval Score"""

import numpy as np
import pandas as pd
from nose.tools import assert_true

from probabilistic_regression_tools.scores.intervalscore import intervalscore
from probabilistic_regression_tools.scores.quantilescore import quantilescore


class Test_Intervalscore:
    """ Test Class for Interval score. """

    def test_intervalscore(self):
        """ Test of interval score, comparison with reference implemenation."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        ypred = df.iloc[:, 0:9]
        ypred = ypred.as_matrix().tolist()
        y = df.iloc[:, 9]

        quantiles = np.linspace(0.1, 0.9, 9)

        overall_is, individual_is = intervalscore(y, ypred, quantiles=quantiles)

        is_good = np.isclose(overall_is, 0.3193, atol=10e-2)
        assert_true(is_good, msg="Value is different to the reference implementation value")

    def test_relationship_quantilescore(self):
        """Test of intervalscore, relationship with quantile score tested."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        ypred = df.iloc[:, 0:9]
        ypred = ypred.as_matrix().tolist()
        y = df.iloc[:, 9]

        quantiles = np.linspace(0.1, 0.9, 9)

        overall_is, individual_is = intervalscore(y, ypred, quantiles=quantiles)

        # Test with relationship to quantilescore
        qs, qs_individual = quantilescore(y, ypred, quantiles=quantiles)
        nr_intervals = int(quantiles.size / 2)
        alphas = 2 * quantiles[0:nr_intervals]

        qs_intervals = [[] for i in range(alphas.size)]
        for i in range(alphas.size):
            qs_intervals[i] = (2 / alphas[i]) * (qs_individual[:, i] + qs_individual[:, quantiles.size - i - 1])

        qs_overall = np.mean(np.mean(qs_intervals))

        is_good = np.isclose(overall_is, qs_overall)
        assert_true(is_good, msg="Equivalence to quantile score not given.")
