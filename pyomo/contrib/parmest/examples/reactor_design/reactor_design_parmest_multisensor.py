import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from reactor_design import reactor_design_model

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'k3']

# Data, includes multiple sensors for ca and cc
data = pd.read_excel('reactor_data_multisensor.xlsx')  

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

pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling

np.random.seed(38256)
bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

parmest.pairwise_plot(bootstrap_theta)
parmest.pairwise_plot(bootstrap_theta, theta, 'rectangular', 0.8)
mvn_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'multivariate_normal', 0.8)
kde_dist = parmest.pairwise_plot(bootstrap_theta, theta, 'gaussian_kde', 0.8)

### Likelihood ratio test

theta_vals = pd.DataFrame(columns=theta_names) # Only vary k2 in this example
for i, k2 in enumerate(np.arange(1.3, 2, 0.1)):
    theta_vals.loc[i,:] = [0.83, k2, 0.00016]
obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta)
LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

LR80 = LR.loc[LR[0.8] == True, theta_names]
parmest.pairwise_plot(LR80)
