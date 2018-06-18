from probabilistic_regression_tools.probmdl import QuantileForecastWrapper
from probabilistic_regression_tools.scores import crps_for_quantiles
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class test_quantile_forecast_wrapper:
    
    def __init__(self):
        self.rd_size = np.random.randint(80,120)
        def sigmoid(x, noise=0.2):
            return (1/(1+np.exp(-x)) + np.random.uniform(size=x.shape)*noise).ravel()*5
        self.x = np.random.uniform(low=-10, high=10, size=(self.rd_size,1))
        self.y = sigmoid(self.x)
        gbr = GradientBoostingRegressor(loss='quantile').fit(self.x,self.y)
        self.qf = QuantileForecastWrapper(gbr).fit(self.x,self.y)
    
    def test_shape(self):
        # Assert dists objs from prediction for every datapoint
        assert len(self.qf.predict_proba(self.x)) == self.rd_size
        
    def test_prediction(self):
        # Assert prediction on fitted data very good
        assert np.mean(crps_for_quantiles(self.qf.predict_proba(self.x),
                                          self.y,self.qf._quantiles_including_zero_and_one_quantile_padding())[0])<0.15