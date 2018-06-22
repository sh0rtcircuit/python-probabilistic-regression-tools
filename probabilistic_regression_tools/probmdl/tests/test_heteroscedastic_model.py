from probabilistic_regression_tools.probmdl.heteroscedastic_model import HeteroscedasticRegression
from probabilistic_regression_tools.scores.crps import crps
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class test_heteroscedastic_model:
    
    def __init__(self):
        self.rd_size = np.random.randint(80,120)
        def sigmoid(x, noise=0.2):
            return (1/(1+np.exp(-x)) + np.random.uniform(size=x.shape)*noise).ravel()*5
        self.x = np.random.uniform(low=-10, high=10, size=(self.rd_size,1))
        self.y = sigmoid(self.x)
        gbr = GradientBoostingRegressor().fit(self.x,self.y)
        self.het = HeteroscedasticRegression(gbr).fit(self.x,self.y)
    
    def test_shape(self):
        # Assert dists objs from prediction for every datapoint
        assert len(self.het.predict(self.x)) == self.rd_size
        
    def test_prediction(self):
        # Assert prediction on fitted data very good
        assert np.mean(crps(self.y, self.het.predict(self.x))[0])<0.15