"""Test for CRPS"""

import numpy as np
import pandas as pd
from nose.tools import assert_true
from scipy.stats import norm
from sklearn import linear_model

import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles
import probabilistic_regression_tools.probmdl.Homoscedastic_Mdl as Homoscedastic_Mdl
import probabilistic_regression_tools.scores.crps as crps
import probabilistic_regression_tools.scores.crps_for_quantiles as crps_for_quantiles


class Test_CRPS:
    """Test class for CRPS."""

    def test_compare_with_closed_form(self):
        """Test that compares the computed with the analytical CRPS."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]

        mean_crps, single_crps = crps(pd, meas)

        def crps_closed_form(pd, meas):
            return meas * (2 * pd.cdf(meas) - 1) + 2 * pd.pdf(meas) - 1 / np.sqrt(np.pi)

        crps_analytical = list(map(crps_closed_form, pd, meas))

        is_good = np.isclose(np.array(single_crps), np.array(crps_analytical)).all()
        assert_true(is_good, msg="Computed CRPS is not equal to analytical CRPS.")

    def test_asymptotical_correctness(self):
        """Tests whether the non-decomposition version and the decomposed version are rougly equal."""
        dfbig = pd.read_csv('probabilistic_regression_tools/scores/tests/wf3.csv')
        df = dfbig.iloc[0:100, :]
        X = df.drop(['Time', 'ForecastingTime', 'PowerGeneration'], axis=1)
        y = df['PowerGeneration']

        mdl = linear_model.LinearRegression()
        mdl.fit(X, y)

        prob_mdl = Homoscedastic_Mdl.Homoscedastic_Mdl(mdl)
        prob_mdl.fit(X, y)

        ypred = prob_mdl.predict(X)
        quantile_vals = np.linspace(0.01, 0.99, 99)

        quantile_forecasts = probdists_2_quantiles.probdists_2_quantiles(ypred, quantiles=quantile_vals)

        crpsv1, _ = crps(ypred, y)
        crpsv2, _ = crps_for_quantiles.crps_for_quantiles(quantile_forecasts, y.as_matrix(), quantiles=quantile_vals)

        isgood = np.isclose(crpsv1, crpsv2, rtol=0.05)
        assert_true(isgood, msg="crps variants are asymptotically not the same.")

    def test_compare_different_expectations(self):
        """Test that compares same distance between meas and pd."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]
        mean_crps1, single_crps1 = crps(pd, meas)

        pd2 = []
        for i in range(0, 3):
            pd2.append(norm(i, 1))
        meas2 = [-1, 1, 3]

        mean_crps2, single_crps2 = crps(pd2, meas2)

        is_good = np.equal(single_crps1, single_crps2).all()
        assert_true(is_good, msg="Relation of individual CPRS values should return same value.")
