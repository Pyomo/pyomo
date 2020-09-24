import numpy as np
import pandas as pd
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model

### Parameter estimation

# Vars to estimate
theta_names = ['asymptote', 'rate_constant']

# Data
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                          [4,16.0],[5,15.6],[7,19.8]],
                    columns=['hour', 'y'])

# Sum of squared error function
def SSE(model, data):  
    expr = sum((data.y[i]\
                - model.response_function[data.hour[i]])**2 for i in data.index)
    return expr

pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE,
                         solver_options=None)
obj, theta, var_values = pest.theta_est(return_values=['response_function'])
print(obj)
print(theta)
print(var_values)
