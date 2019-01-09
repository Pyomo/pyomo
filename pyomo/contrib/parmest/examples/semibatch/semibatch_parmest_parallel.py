"""
The following script can be run semibatch parameter estimation in parallel 
and save results to files for later analysis.
Example command: mpiexec -n 4 python semibatch_parmest_parallel.py
"""
import numpy as np
import pandas as pd
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

theta_vals = pd.DataFrame(columns=theta_names)
i = 0
for k1 in np.arange(4, 24, 3):
    for k2 in np.arange(40, 160, 40):
        for E1 in np.arange(29000, 32000, 500):
            for E2 in np.arange(38000, 42000, 500):
                theta_vals.loc[i,:] = [k1, k2, E1, E2]
                i = i+1
LR = pest.objective_at_theta(theta_vals)
LR.to_csv('LR.csv')
