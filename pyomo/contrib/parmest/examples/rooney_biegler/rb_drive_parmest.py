# This file calls parmest and makes use of rooney_biegler.py to provide the 
# model and callback.

import numpy as np
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
import pyomo.contrib.parmest.mpi_utils as mpiu

# prepare for the parmest object construction
experiment_data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                      [4,16.0],[5,15.6],[6,19.8]],
                                      columns=['hour', 'y'])
num_samples = 6
sample_list = list(range(num_samples)) # callback uses zero-based indexes
thetalist = ['asymptote', 'rate_constant']
fsfilename = "rooney_biegler"

np.random.seed(1134)

# Generate parmest object
pest = parmest.ParmEstimator(fsfilename, "instance_creation_callback",
                             "SecondStageCost", sample_list, thetalist,
                             cb_data = experiment_data)

mpii = mpiu.MPIInterface() # works with, or without, mpi

### Parameter estimation with entire data set
objval, thetavals = pest.theta_est()

if mpii.rank in [0, None]:
    print ("objective value=",str(objval))
    print ("theta-star=",str(thetavals))
  
### Parameter estimation with bootstrap
alpha = 0.8
num_bootstraps = 10
bootstrap_theta = pest.bootstrap(num_bootstraps)
if mpii.rank in [0, None]:
    print ("Bootstrap:")
    print(bootstrap_theta)
    grph.pairwise_plot(bootstrap_theta, filename="RB.png")
    grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, alpha, 
                                 filename="RB_boot.png")

### Likelihood ratio
alpha = 0.8
search_ranges = {}
search_ranges["asymptote"] = np.arange(10, 30, 2) # np.arange(10, 30, 0.01)
search_ranges["rate_constant"] = np.arange(0, 1.5, 0.1) # np.arange(0, 1.5, 0.005)
SSE = pest.likelihood_ratio(search_ranges=search_ranges)
if mpii.rank in [0, None]:
    print ("Likelihood Ratio:")
    print(SSE)
    grph.pairwise_likelihood_ratio_plot(SSE, objval, alpha, num_samples, 
                                        filename="RB_LR.png")

