import numpy as np
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
import pyomo.contrib.parmest.mpi_utils as mpiu
from semibatch import generate_model

mpii = mpiu.MPIInterface() # works with, or without, mpi

### Parameter estimation

# Vars to estimate
thetavars = ['k1', 'k2', 'E1', 'E2']

# Data, list of json file names
data = [] 
for exp_num in range(10):
    data.append('exp'+str(exp_num+1)+'.out')

# Note, the model already includes a 'SecondStageCost' expression 
# for sum of squared error that will be used in parameter estimation
        
pest = parmest.Estimator(generate_model, data, thetavars)
obj, theta = pest.theta_est()
if mpii.rank in [0, None]:
    print(obj)
    print(theta)


### Parameter estimation with bootstrap resampling

np.random.seed(38256)
bootstrap_theta = pest.bootstrap(50)
if mpii.rank in [0, None]:
    print(bootstrap_theta.head())
    grph.pairwise_bootstrap_plot(bootstrap_theta, theta, 0.8)


### Parameter estimation with likelihood ratio

search_ranges = {}
search_ranges["E1"] = np.arange(29000, 32000, 1000)
search_ranges["E2"] = np.arange(38000, 42000, 1000) 
search_ranges["k1"] = np.arange(4, 24, 6) 
search_ranges["k2"] = np.arange(40, 160, 80)
LR = pest.likelihood_ratio(search_ranges=search_ranges)
if mpii.rank in [0, None]:
    print(LR.head())
    grph.pairwise_likelihood_ratio_plot(LR, obj, 0.8, len(data))
