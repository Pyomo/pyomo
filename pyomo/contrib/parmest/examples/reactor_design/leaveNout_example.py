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
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model 

### Parameter estimation

# Vars to estimate
theta_names = ['k1', 'k2', 'k3']

# Data
data = pd.read_excel('reactor_data.xlsx') 

# Create more data for the example
df_std = data.std().to_frame().transpose()
df_rand = pd.DataFrame(np.random.normal(size=100))
df_sample = data.sample(100, replace=True).reset_index(drop=True)
data = df_sample + df_rand.dot(df_std)/10

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

### Parameter estimation with 'leave-N-out'
# Example use case: For each combination of data where one data point is left 
# out, estimate theta
lNo_theta = pest.theta_est_leaveNout(1) 
print(lNo_theta.head())

parmest.graphics.pairwise_plot(lNo_theta, theta)

### Leave one out/boostrap analysis
# Example use case: leave 50 data points out, run 75 bootstrap samples with the 
# remaining points, determine if the theta estimate using the points left out 
# is inside or outside an alpha region based on the bootstrap samples, repeat 
# 10 times. Results are stored as a list of tuples, see API docs for information.
lNo = 50
lNo_samples = 10
bootstrap_samples = 75
dist = 'MVN'
alphas = [0.7, 0.8, 0.9]

results = pest.leaveNout_bootstrap_test(lNo, lNo_samples, bootstrap_samples, 
                                            dist, alphas, seed=524)

# Plot results for a single value of alpha
alpha = 0.8
for i in range(lNo_samples):
    theta_est_N = results[i][1]
    bootstrap_results = results[i][2]
    parmest.graphics.pairwise_plot(bootstrap_results, theta_est_N, alpha, ['MVN'],
                          title= 'Alpha: '+ str(alpha) + ', '+ \
                          str(theta_est_N.loc[0,alpha]))
    
# Extract the percent of points that are within the alpha region
r = [results[i][1].loc[0,alpha] for i in range(lNo_samples)]
percent_true = sum(r)/len(r)
print(percent_true)
