import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from rooney_biegler import rooney_biegler_model

### Parameter estimation

# Vars to estimate
theta_names = ['asymptote', 'rate_constant']

# Data
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                          [4,16.0],[5,15.6],[6,19.8]],
                    columns=['hour', 'y'])

# Sum of squared error function
def SSE(model, data):  
    expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
    return expr

pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling

np.random.seed(4581)
bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

parmest.pairwise_plot(bootstrap_theta, theta, 'rectangular', 0.8)
mvn_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'multivariate_normal', 0.8)
kde_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'gaussian_kde', 0.8)

### Parameter estimation with likelihood ratio

theta_vals = pd.DataFrame(columns=theta_names)
i = 0
for asym in np.arange(10, 30, 2):
    for rate in np.arange(0, 1.5, 0.1):
        theta_vals.loc[i,:] = [asym, rate]
        i = i+1
obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta.head())
LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

LR80 = LR.loc[LR[0.8] == True, theta_names]
parmest.pairwise_plot(LR80)
