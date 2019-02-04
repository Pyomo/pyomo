import numpy as np
import pandas as pd
from itertools import product
import json
import pyomo.contrib.parmest.parmest as parmest
from semibatch import generate_model

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'E1', 'E2']

# Data, list of dictionaries
data = [] 
for exp_num in range(10):
    fname = 'exp'+str(exp_num+1)+'.out'
    with open(fname,'r') as infile:
        d = json.load(infile)
        data.append(d)

# Note, the model already includes a 'SecondStageCost' expression 
# for sum of squared error that will be used in parameter estimation
        
pest = parmest.Estimator(generate_model, data, theta_names)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling

bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

parmest.pairwise_plot(bootstrap_theta, title='Bootstrap theta estimates')
parmest.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'], 
                      title='Bootstrap theta with confidence regions')

### Parameter estimation with leave one out (LOO)

LOO_theta = pest.theta_est_leaveNout(1)
print(LOO_theta.head())

parmest.pairwise_plot(LOO_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'], 
                      title='LOO results with confidence regions')

LOO_test_results = pest.alpha_test(LOO_theta, 'MVN', [0.8, 0.85, 0.9, 0.95])
print(LOO_test_results.head())

parmest.pairwise_plot(LOO_test_results, theta, 0.8, 
                      title='LOO results within 80% confidence region')

### Likelihood ratio test

k1 = [19]
k2 = np.arange(40, 160, 40)
E1 = [30524]
E2 = np.arange(38000, 42000, 500)
theta_vals = pd.DataFrame(list(product(k1, k2, E1, E2)), columns=theta_names)

obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta.head())

LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

theta_slice = {'k1': 19, 'k2': theta['k2'], 'E1': 30524, 'E2': theta['E2']}
parmest.pairwise_plot(LR, theta_slice, 0.8, 
                      title='LR results within 80% confidence region')
