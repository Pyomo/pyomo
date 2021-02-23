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
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model 

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

bootstrap_theta = pest.theta_est_bootstrap(50)
print(bootstrap_theta.head())

parmest.graphics.pairwise_plot(bootstrap_theta, title='Bootstrap theta estimates')
parmest.graphics.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'], 
                      title='Bootstrap theta with confidence regions')

### Likelihood ratio test

k1 = np.arange(0.78, 0.92, 0.02)
k2 = np.arange(1.48, 1.79, 0.05)
k3 = np.arange(0.000155, 0.000185, 0.000005)
theta_vals = pd.DataFrame(list(product(k1, k2, k3)), columns=theta_names)

obj_at_theta = pest.objective_at_theta(theta_vals)
print(obj_at_theta.head())

LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
print(LR.head())

parmest.graphics.pairwise_plot(LR, theta, 0.8, 
                      title='LR results within 80% confidence region')
