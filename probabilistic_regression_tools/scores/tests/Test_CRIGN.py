"""Test for CRIGN"""

import numpy as np
import pandas as pd
import probabilistic_regression_tools.probdists_2_quantiles as probdists_2_quantiles
import probabilistic_regression_tools.probmdl.Homoscedastic_Mdl as Homoscedastic_Mdl
import scipy.integrate as integrate
from nose.tools import assert_equal, assert_true
from scipy.stats import norm
from sklearn import linear_model

import probabilistic_regression_tools.scores.crign as crign
import probabilistic_regression_tools.scores.crign_for_quantiles as crign_for_quantiles


class Test_CRIGN:
    """Test class for CRIGN."""

    def test_check_result_with_expectation(self):
        """Checks the resulting value with an expectation value."""
        pd_single = norm(0, 1)
        pd = [pd_single]
        meas = [-1]

        [meancrign, singlecrign] = crign.crign(pd, meas)

        # computation by hand
        # integrate from -inf to current measurement
        fixCdfL = lambda x: np.log(np.abs(pd_single.cdf(x) - 1))
        S1 = integrate.quad(fixCdfL, -np.inf, meas[0])

        # integrate from measurement to inf
        fixCdfU = lambda x: np.log(np.abs(pd_single.cdf(x) - 0))
        S2 = integrate.quad(fixCdfU, meas[0], np.inf)

        singlecrign2 = -(S1[0] + S2[0])
        assert_equal(meancrign, singlecrign2, msg="CRIGN values should be equal.")

    def test_asymptotical_correctness(self):
        """Compares the CRIGN variants using quantiles and integration."""

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

        crignv1, _ = crign.crign(ypred, y)
        crignv2, _ = crign_for_quantiles.crign_for_quantiles(quantile_forecasts, y.as_matrix(), quantiles=quantile_vals)

        isgood = np.isclose(crignv1, crignv2, rtol=0.05)
        assert_true(isgood, msg="CRIGN variants are asymptotically not the same.")

    def test_compare_different_expectations(self):
        """Test that compares same distance between meas and pd."""

        pd_single = norm(0, 1)
        pd = []
        for i in range(0, 3):
            pd.append(pd_single)
        meas = [-1, 0, 1]
        meanCRIGN1, singleCRIGN1 = crign.crign(pd, meas)

        pd2 = []
        for i in range(0, 3):
            pd2.append(norm(i, 1))
        meas2 = [-1, 1, 3]

        meanCRIGN2, singleCRIGN2 = crign.crign(pd2, meas2)

        is_good = np.isclose(singleCRIGN1, singleCRIGN2).all()
        assert_true(is_good, msg="Relation of individual CRIGN values should return roughly the same value.")
