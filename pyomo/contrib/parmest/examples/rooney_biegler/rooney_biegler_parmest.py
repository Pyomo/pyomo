import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
from rooney_biegler import rooney_biegler_model

### Parameter estimation

# Vars to estimate
thetavars = ['asymptote', 'rate_constant']

# Data
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                          [4,16.0],[5,15.6],[6,19.8]],
                    columns=['hour', 'y'])

# Sum of squared error function
def SSE(model, data):  
    expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
    return expr

pest = parmest.Estimator(rooney_biegler_model, data, thetavars, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)


### Parameter estimation with bootstrap resampling

np.random.seed(4581)
bootstrap_theta = pest.bootstrap(100)
print(bootstrap_theta.head())
grph.pairwise_bootstrap_plot(bootstrap_theta, 0.8, theta)

### Parameter estimation with likelihood ratio

search_ranges = {}
search_ranges["asymptote"] = np.arange(10, 30, 2) 
search_ranges["rate_constant"] = np.arange(0, 1.5, 0.1) 
LR = pest.likelihood_ratio(search_ranges=search_ranges)
print(LR.head())
grph.pairwise_likelihood_ratio_plot(LR, obj, 0.8, data.shape[0], theta)
