"""
The following script can be run semibatch parameter estimation in parallel 
and save results to files for later analysis.
Example command: mpiexec -n 4 python semibatch_parmest_parallel.py
"""
import numpy as np
import pyomo.contrib.parmest.parmest as parmest
from semibatch import generate_model

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'E1', 'E2']

# Data, list of json file names
data = [] 
for exp_num in range(10):
    data.append('exp'+str(exp_num+1)+'.out')

# Note, the model already includes a 'SecondStageCost' expression 
# for sum of squared error that will be used in parameter estimation
        
pest = parmest.Estimator(generate_model, data, theta_names)

### Parameter estimation with bootstrap resampling

np.random.seed(38256)
bootstrap_theta = pest.theta_est_bootstrap(100)
bootstrap_theta.to_csv('bootstrap_theta.csv')


### Compute objective at theta for likelihood ratio test

search_ranges = {}
search_ranges["E1"] = np.arange(29000, 32000, 500)
search_ranges["E2"] = np.arange(38000, 42000, 500) 
search_ranges["k1"] = np.arange(4, 24, 3) 
search_ranges["k2"] = np.arange(40, 160, 40)
LR = pest.objective_at_theta(search_ranges=search_ranges)
LR.to_csv('LR.csv')
