# This file calls parmest and makes use of semibatch.py to provide the 
# model and callback.  Data is supplied to the through files in the callback.

import numpy as np
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph
import pyomo.contrib.parmest.mpi_utils as mpiu

num_experiments = 10
exp_list = range(1,num_experiments+1) # callback uses one-based file names
thetalist = ['k1', 'k2', 'E1', 'E2']
fsfilename = "semibatch"

np.random.seed(42)

pest = parmest.ParmEstimator(fsfilename, "pysp_instance_creation_callback",
                             "SecondStageCost", exp_list, thetalist)

objval, thetavals = pest.theta_est()

mpii = mpiu.MPIInterface() # works with, or without, mpi

if mpii.rank in [0, None]:
    print ("objective value=",str(objval))
    print ("theta-star=",str(thetavals))

### Parameter estimation with bootstrap
num_bootstraps = 10
bootstrap_theta = pest.bootstrap(num_bootstraps)
# generate the output, but only if on main processor
if mpii.rank in [0, None]:
    print ("Bootstrap:")
    print(bootstrap_theta)
    grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, 
                                 filename="SB_boot.png")

### Make the LR plot
### special hard-wired search ranges for debugging
search_ranges = {}
search_ranges["E1"] = np.arange(29000, 32000, 1000)
search_ranges["E2"] = np.arange(38000, 42000, 1000) 
search_ranges["k1"] = np.arange(4, 24, 6) 
search_ranges["k2"] = np.arange(40, 160, 80)
SSE = pest.likelihood_ratio(search_ranges=search_ranges)
print(SSE)
if mpii.rank in [0, None]:
    print ("Likelihood Ratio:")
    print(SSE)
    grph.pairwise_likelihood_ratio_plot(SSE, objval, 0.8, num_experiments, 
                                        filename="SB_LR.png")
