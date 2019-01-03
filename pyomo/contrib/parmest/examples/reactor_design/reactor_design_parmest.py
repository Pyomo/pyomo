import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
from reactor_design import reactor_design_model

### Parameter estimation

# Vars to estimate
thetavars = ['k1', 'k2', 'k3']

# Data, includes multiple sensors for ca and cc
data = pd.read_excel('reactor_data.xlsx')  

# Sum of squared error function
def SSE(model, data): 
    expr = ((float(data['ca1']) - model.ca)**2)*(1/3) + \
           ((float(data['ca2']) - model.ca)**2)*(1/3) + \
           ((float(data['ca3']) - model.ca)**2)*(1/3) + \
            (float(data['cb'])  - model.cb)**2 + \
           ((float(data['cc1']) - model.cc)**2)*(1/2) + \
           ((float(data['cc2']) - model.cc)**2)*(1/2) + \
            (float(data['cd'])  - model.cd)**2
    return expr

pest = parmest.Estimator(reactor_design_model, data, thetavars, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling

np.random.seed(38256)
bootstrap_theta = pest.bootstrap(50)
print(bootstrap_theta.head())
grph.pairwise_plot(bootstrap_theta, theta)
grph.pairwise_bootstrap_plot(bootstrap_theta, 0.8, theta)

### Parameter estimation with likelihood ratio

search_ranges = {}
search_ranges['k1'] = np.arange(0.78, 0.92, 0.02) 
search_ranges['k2'] = np.arange(1.48, 1.79, 0.05) 
search_ranges['k3'] = np.arange(0.000155, 0.000185, 0.000005) 
LR = pest.likelihood_ratio(search_ranges=search_ranges)
print(LR.head())
grph.pairwise_likelihood_ratio_plot(LR, obj, 0.8, data.shape[0], theta)