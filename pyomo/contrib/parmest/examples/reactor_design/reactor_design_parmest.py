import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from reactor_design import reactor_design_model

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'k3']

# Data
data = pd.read_excel('reactor_data.xlsx') 

# Sum of squared error function
def SSE(model, data): 
    expr = (float(data['ca']) - model.ca)**2 + \
           (float(data['cb']) - model.cb)**2 + \
           (float(data['cc']) - model.cc)**2 + \
           (float(data['cd']) - model.cd)**2
    return expr

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling

np.random.seed(12524)
bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

parmest.pairwise_plot(bootstrap_theta)
parmest.pairwise_plot(bootstrap_theta, theta, 'rectangular', 0.8)
mvn_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'multivariate_normal', 0.8)
kde_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'gaussian_kde', 0.8)

### Likelihood ratio test

theta_vals = pd.DataFrame(columns=theta_names)
i = 0
for k1 in np.arange(0.78, 0.92, 0.02):
    for k2 in np.arange(1.48, 1.79, 0.05):
        for k3 in np.arange(0.000155, 0.000185, 0.000005):
            theta_vals.loc[i,:] = [k1, k2, k3]
            i = i+1
obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta)
LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

LR80 = LR.loc[LR[0.8] == True, theta_names]
parmest.pairwise_plot(LR80)
