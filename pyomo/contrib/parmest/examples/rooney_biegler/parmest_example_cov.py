#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model

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
obj, theta, cov = pest.theta_est(calc_cov=True)
print(obj)
print(theta)
print(cov)

parmest.graphics.pairwise_plot((theta, cov, 1000), theta_star=theta, alpha=0.8, 
                               distributions=['MVN'])
