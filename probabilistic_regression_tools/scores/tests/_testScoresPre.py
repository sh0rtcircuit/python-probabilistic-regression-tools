# -*- coding: utf-8 -*-

# this will be the CRPS test
#from probabilistic_regression_tools.scores.crps import crps
#from probabilistic_regression_tools.scores.quantilescore import quantilescore
from probabilistic_regression_tools.scores.tests.Test_CRPS import Test_CRPS
from probabilistic_regression_tools.scores.tests.Test_Quantilescore import Test_Quantilescore
#from probabilistic_regression_tools.scores.tests.testIgnorance import Test_Ignorance
from probabilistic_regression_tools.scores.tests.Test_CRIGN import Test_CRIGN
#from probabilistic_regression_tools.scores.tests.test_crps_with_decompostition import Test_CRPS_With_Decomposition
from probabilistic_regression_tools.scores.tests.Test_CRIGN_With_Decomposition import Test_CRIGN_With_Decomposition
from probabilistic_regression_tools.scores.crign_for_quantiles import crign_for_quantiles
#from probabilistic_regression_tools.tests.test_probdists_2_quantiles import Test_Probdists_2_Quantiles
from probabilistic_regression_tools.scores.tests.Test_Quantilescore_With_Decomposition import Test_Quantilescore_With_Decomposition
from probabilistic_regression_tools.scores.tests.Test_Intervalscore import Test_Intervalscore
from probabilistic_regression_tools.scores.tests.Test_Sphericalscore import Test_Sphericalscore
from probabilistic_regression_tools.scores.tests.Test_Quadraticscore import Test_Quadraticscore
from probabilistic_regression_tools.scores.tests.Test_DawidSebastianiscore import Test_DawidSebastianiscore
import probabilistic_regression_tools.probmdl.Heteroscedastic_Mdl as Heteroscedastic_Mdl
import numpy as np
import pandas as pd
from scipy.stats import norm
import sklearn.linear_model as linear_model

#Test_CRPS.test_compare_different_expectations(0)
#
#pdSingle = norm(0, 1)
#pd = []
#for i in range(0,3):
#    pd.append(pdSingle)
#meas = [-1, 0, 1]
#
#meanCRPS, singleCRPS = crps(pd, meas)
#
## compare with closed form solution
## from https://cran.r-project.org/web/packages/scoringRules/vignettes/crpsformulas.html#Normal
#def crpsClosedForm (pd,meas): return meas * (2 * pd.cdf(meas) -1) + 2 * pd.pdf(meas) - 1./np.sqrt(np.pi)
##obj.crpsAnalytical = cellfun(crpsClosedForm,obj.probForecastsNorm,num2cell(obj.measurements));
#
#print(meanCRPS)
#print(singleCRPS)

#qsdd = probabilistic_regression_tools.scores.quantilescore.quantilescore(quantileForecasts,meas,quantileValues=np.linspace(0.1,0.9,9))
#
#quantiles = np.linspace(0.1,0.9,9)
#quantileForecast = np.linspace(-0.4,0.4,9)
#
#Test_Quantilescore.test_with_quantiles(0)
#Test_QuantileScore.test_with_distribution(0)
#Test_QuantileScore.test_with_single_tau(0)

#Test_Ignorance.test_ignorance(0)
#Test_CRIGN.test_compare_different_expectations(0)
##Test_CRIGN.test_checkResult(0)
#if __name__ == "__main__":
#    Test_CRPS_With_Decomposition.compare_with_reference_implementation(0)
#    Test_CRPS_With_Decomposition.compareDeviationWithConventionalCrps(0)

#Test_Probdists_2_Quantiles.test_probdists_2_quantiles(0)
#Test_CRIGN_With_Decomposition.test_compare_with_reference_implementation(0)


#Test_CRIGN_With_Decomposition.test_compare_deviation_with_conventional_crign(0)
#Test_CRIGN_With_Decomposition.test_compare_with_non_decomposition_version(0)
#Test_CRIGN.test_asymptotical_correctness(0)
#Test_CRIGN.test_checkResult(0)

#Test_Quantilescore_With_Decomposition.test_compare_with_reference_implementation(0)
#Test_Quantilescore_With_Decomposition.test_compare_deviation_with_conventional_qs(0)
#Test_Intervalscore.test_intervalscore(0)
#myobj = Test_DawidSebastianiscore()
#myobj.test_compare_different_expectations()

#Test_CRPS.test_asymptotical_correctness(0)

#crignval = crign_for_quantiles(quantileForecasts,y)
#Test_CRIGN.test_asymptotical_correctness(0)
#crigndebug = 1

dfbig = pd.read_csv('probabilistic_regression_tools/scores/tests/wf3.csv')
df = dfbig.iloc[0:100, :]
X = df.drop(['Time', 'ForecastingTime', 'PowerGeneration'], axis=1)
y = df['PowerGeneration']

mdl = linear_model.LinearRegression()
mdl.fit(X, y)

prob_mdl = Heteroscedastic_Mdl.Heteroscedastic_Mdl(mdl)
prob_mdl.fit(X, y)

ypred = prob_mdl.predict(X)


