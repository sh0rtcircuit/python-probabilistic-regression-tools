"""Test for CRIGN with Decomposition"""

import numpy as np
import pandas as pd
from nose.tools import assert_true

import probabilistic_regression_tools.scores.crign_for_quantiles as crign_for_quantiles
import probabilistic_regression_tools.scores.crign_with_decomposition as crign_with_decomposition


class Test_CRIGN_With_Decomposition:
    """Test class for CRIGN with decomposition."""

    def test_compare_with_reference_implementation(self):
        """Compares the realization with expectations from a matlab reference implementation."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        ypred = df.iloc[:, 0:9]
        ypred = ypred.as_matrix().tolist()
        y = df.iloc[:, 9]

        [crign_all, crign_rel, crign_res, crign_unc] = crign_with_decomposition.crign_with_decomposition(ypred, y)
        print(crign_all)
        print(crign_rel)
        print(crign_res)
        print(crign_unc)

        # from reference implementation
        crign_all_ref = 0.3073
        crign_rel_ref = 0.1504
        crign_res_ref = 0.2432
        crign_unc_ref = 0.4001

        is_good = np.isclose([crign_all, crign_rel, crign_res, crign_unc],
                             [crign_all_ref, crign_rel_ref, crign_res_ref, crign_unc_ref], atol=1e-03).all()
        assert_true(is_good, msg="Result does not match reference implemenation.")

    def test_compare_deviation_with_conventional_crign(self):
        """Tests whether the non-decomposition version and the decomposed version are rougly equal."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc2.csv')
        quantile_forecasts = df.drop(['y'], axis=1).as_matrix()
        y = df['y'].as_matrix()

        quantile_vals = np.linspace(0.01, 0.99, 99)

        # compute both variants
        crignval, _ = crign_for_quantiles.crign_for_quantiles(quantile_forecasts, y, quantiles=quantile_vals)
        [crign_all, crign_rel, crign_res, crign_unc] = crign_with_decomposition.crign_with_decomposition(
            quantile_forecasts, y,
            quantiles=quantile_vals)

        # relatively large deviation due to slightly different computation (9 quantiles vs. entire cdf)
        is_good = np.abs(crign_all - crignval) <= 0.2 * crignval
        assert_true(is_good, msg="Decomposition error larger than expected.")
