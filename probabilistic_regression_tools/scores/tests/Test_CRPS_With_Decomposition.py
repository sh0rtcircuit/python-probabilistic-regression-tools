"""Test for CRPS with Decomposition"""

import numpy as np
import pandas as pd
from probabilistic_regression_tools.probmdl.homoscedastic_model import HomoscedasticRegression
from nose.tools import assert_true
from sklearn import linear_model

import probabilistic_regression_tools.scores.crps as crps
import probabilistic_regression_tools.scores.crps_with_decomposition as crps_with_decomposition


class Test_CRPS_With_Decomposition:
    """Test class for CRPS with decomposition."""

    def test_compare_with_reference_implementation(self):
        """Compares the realization with expectations from a matlab reference implementation."""

        df = pd.read_csv('probabilistic_regression_tools/scores/tests/probFc.csv')
        ypred = df.iloc[:, 0:9]
        ypred = ypred.as_matrix().tolist()
        y = df.iloc[:, 9]

        [crps_all, crps_rel, crps_res, crps_unc] = crps_with_decomposition.crps_with_decomposition(ypred, y)
        print(crps_all)
        print(crps_rel)
        print(crps_res)
        print(crps_unc)

        # from reference implementation
        crps_all_ref = 0.0775
        crps_rel_ref = 0.0324
        crps_res_ref = 0.0851
        crps_unc_ref = 0.1303

        is_good = np.isclose([crps_all, crps_rel, crps_res, crps_unc],
                             [crps_all_ref, crps_rel_ref, crps_res_ref, crps_unc_ref],
                             atol=1e-03).all()
        assert_true(is_good, msg="Result does not match reference implemenation.")

    def test_compare_deviation_with_conventional_crps(self):
        """Tests whether the non-decomposition version and the decomposed version are rougly equal."""

        dfbig = pd.read_csv('probabilistic_regression_tools/scores/tests/wf3.csv')
        df = dfbig.iloc[0:100, :]
        X = df.drop(['Time', 'ForecastingTime', 'PowerGeneration'], axis=1)
        y = df['PowerGeneration']

        mdl = linear_model.LinearRegression()
        mdl.fit(X, y)

        prob_mdl = HomoscedasticRegression(mdl)
        prob_mdl.fit(X, y)

        ypred = prob_mdl.predict(X)

        [crps_all, crps_rel, crps_res, crps_unc] = crps_with_decomposition.crps_with_decomposition(ypred, y)

        crpsval = crps(ypred, y)
        print(crpsval[0])

        # relatively large deviation due to slightly different computation (9 quantiles vs. entire cdf)
        is_good = np.abs(crps_all - crpsval[0]) <= 0.2 * crpsval[0]
        assert_true(is_good, msg="Decomposition error larger than expected.")
