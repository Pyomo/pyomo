# parmest example using mea

import numpy as np
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.mpi_utils as mpiu
import pyomo.contrib.parmest.graphics as grph

if __name__ == "__main__":

    thetalist = ['pp.Keq_a[1]', 'pp.Keq_a[2]']
    S = 4

    fsfilename = "examples.contrib.projects.mea_simple.parameter_estimate.mea_estimate_pysp"

    pest = parmest.ParmEstimator(fsfilename,
                                  "pysp_instance_creation_callback",
                                 "SecondStageCost",
                                 range(1,S+1),
                                 thetalist)

    mpii = mpiu.MPIInterface() # works with, or without, mpi

    objval, thetavals = pest.theta_est()
    if mpii.rank in [0, None]:
        print ("objective value=",str(objval))
        for t in thetavals:
            print (t, "=", thetavals[t])
        print ("====")

    ### Parameter estimation with bootstrap
    np.random.seed(1134) # make it reproducible
    num_bootstraps = 10
    bootstrap_theta = pest.bootstrap(num_bootstraps)
    if mpii.rank in [0, None]:
        print ("Bootstrap:")
        print(bootstrap_theta)
        grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, 
                                     filename="MEA_boot.png")

    ### Make the LR plot
    # SEE = pest.likelihood_ratio(0.8)
    # grph.pairwise_likelihood_ratio_plot(SSE, objval, 0.8, S, "MEA_LR.png")
