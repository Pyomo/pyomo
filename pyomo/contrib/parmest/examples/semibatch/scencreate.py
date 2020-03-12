# scenario creation; DLW March 2020
import numpy as np
import pandas as pd
from itertools import product
import json
import pyomo.contrib.parmest.parmest as parmest
from semibatch import generate_model
import pyomo.environ as pyo

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

# create one scenario for each experiment
for exp_num in pest._numbers_list:
    print("Experiment number=", exp_num)
    model = pest._instance_creation_callback(exp_num, data)
    opt = pyo.SolverFactory('ipopt')
    results = opt.solve(model)  # solves and updates model
    ## pyo.check_termination_optimal(results)
    for theta in pest.theta_names:
        tvar = eval('model.'+theta)
        tval = pyo.value(tvar)
        print("    tvar, tval=", tvar, tval)


###obj, theta = pest.theta_est()
###print(obj)
###print(theta)

### Parameter estimation with bootstrap resampling

bootstrap_theta = pest.theta_est_bootstrap(10)
print(bootstrap_theta.head())

###parmest.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'])

