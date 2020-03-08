# scenario creation; DLW March 2020
import numpy as np
import pandas as pd
from itertools import product
import json
import pyomo.contrib.parmest.parmest as parmest
from semibatch import generate_model

# Vars to estimate in parmest
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

###obj, theta = pest.theta_est()
###print(obj)
###print(theta)

### Parameter estimation with bootstrap resampling

bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

###parmest.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'])

