# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
import probabilistic_regression_tools.probdists_2_quantiles
from nose.tools import assert_true

class Test_Probdists_2_Quantiles:
    """Test class for CRPS."""
    
    def test_probdists_2_quantiles(self):
        normdist1 = scipy.stats.norm(0,1)
        normdist2 = scipy.stats.norm(1,2)
        
        normdist_list = [normdist1, normdist2]
        quantiles = np.linspace(0.1,0.9,9)
        
        correctAnswer = np.array([normdist1.ppf(quantiles),normdist2.ppf(quantiles)])
        result = probabilistic_regression_tools.probdists_2_quantiles.probdists_2_quantiles(normdist_list,quantiles)
        
        isGood = np.equal(correctAnswer,result).all()
        assert_true(isGood,msg="Unexpected difference in ppf function.")
        