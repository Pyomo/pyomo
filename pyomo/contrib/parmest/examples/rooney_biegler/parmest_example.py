#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
    expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
    return expr


solver_options = {"max_iter": 6000}  # not really needed in this case

pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE, solver_options)
obj, theta = pest.theta_est()
print(obj)
print(theta)

### Parameter estimation with bootstrap resampling
bootstrap_theta = pest.theta_est_bootstrap(50, seed=4581)
print(bootstrap_theta.head())

parmest.graphics.pairwise_plot(bootstrap_theta, title='Bootstrap theta')
parmest.graphics.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'], 
                      title='Bootstrap theta with confidence regions')

### Likelihood ratio test
asym = np.arange(10, 30, 2)
rate = np.arange(0, 1.5, 0.1)
theta_vals = pd.DataFrame(list(product(asym, rate)), columns=theta_names)

obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta.head())

LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

parmest.graphics.pairwise_plot(LR, theta, 0.8, 
                      title='LR results within 80% confidence region')
