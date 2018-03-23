"""Test for Quantile Score"""

import numpy as np
import pandas as pd
from nose.tools import assert_true

from probabilistic_regression_tools.scores.quantilescore import quantilescore
from probabilistic_regression_tools.scores.quantilescore_with_decomposition import quantilescore_with_decomposition


class Test_Quantilescore_With_Decomposition:
    """ Test Class for quantile score. """

    def test_compare_with_reference_implementation(self):
        """Compares the realization with expectations from a matlab reference implementation."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        ypred = df.iloc[:, 0:9]
        ypred = ypred.as_matrix().tolist()
        y = df.iloc[:, 9]

        [qs_all, qs_rel, qs_res, qs_unc] = quantilescore_with_decomposition(ypred, y)
        print(qs_all)
        print(qs_rel)
        print(qs_res)
        print(qs_unc)

        # from reference implementation
        qs_all_ref = [0.0200, 0.0324, 0.0405, 0.0454, 0.0465, 0.0456, 0.0413, 0.0322, 0.0209]
        qs_rel_ref = [0.0082, 0.0124, 0.0148, 0.0160, 0.0156, 0.0147, 0.0123, 0.0083, 0.0057]
        qs_res_ref = [0.0103, 0.0234, 0.0374, 0.0507, 0.0629, 0.0695, 0.0714, 0.0632, 0.0383]
        qs_unc_ref = [0.0222, 0.0435, 0.0631, 0.0801, 0.0938, 0.1004, 0.1004, 0.0870, 0.0535]

        # check if everything matches
        is_good = np.isclose([qs_all, qs_rel, qs_res, qs_unc], [qs_all_ref, qs_rel_ref, qs_res_ref, qs_unc_ref],
                             atol=1e-03).all()
        assert_true(is_good, msg="Result without bin_averaging does not match reference implemenation.")

        # now with bin averaging
        [qs_all, qs_rel, qs_res, qs_unc] = quantilescore_with_decomposition(ypred, y, bin_averaging=True)

        # from reference implementation
        qs_all_ref = [0.0200, 0.0324, 0.0405, 0.0454, 0.0465, 0.0456, 0.0413, 0.0322, 0.0209]
        qs_rel_ref = [0.0077, 0.0105, 0.0114, 0.0105, 0.0075, 0.0040, -0.0012, -0.0094, -0.0174]
        qs_res_ref = [0.0099, 0.0220, 0.0354, 0.0487, 0.0616, 0.0721, 0.0810, 0.0849, 0.0806]
        qs_unc_ref = [0.0222, 0.0440, 0.0646, 0.0836, 0.1006, 0.1137, 0.1235, 0.1265, 0.1189]

        # check if everything matches
        is_good = np.isclose([qs_all, qs_rel, qs_res, qs_unc], [qs_all_ref, qs_rel_ref, qs_res_ref, qs_unc_ref],
                             atol=1e-03).all()
        assert_true(is_good, msg="Result with quantile averaging does not match reference implemenation.")

    def test_compare_deviation_with_conventional_qs(self):
        """Tests whether the non-decomposition version and the decomposed version are rougly equal."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        quantile_forecasts = df.drop(['y'], axis=1).as_matrix()
        y = df['y'].as_matrix()

        quantile_vals = np.linspace(0.1, 0.9, 9)

        # compute both variants
        qsval, _ = quantilescore(quantile_forecasts, y, quantiles=quantile_vals)
        [qs_all, qs_rel, qs_res, qs_unc] = quantilescore_with_decomposition(quantile_forecasts, y,
                                                                            quantiles=quantile_vals)

        # within 10% tolerance
        is_good = np.isclose(qs_all, qsval, rtol=1e-1).all()
        assert_true(is_good, msg="Decomposition error larger than expected.")
