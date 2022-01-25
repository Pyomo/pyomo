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

def parameter_estimation(pest):
    # Estimate parameter values and return the covariance
    obj, theta, cov = pest.theta_est(calc_cov=True)
    
    # Plot theta estimates using a multivariate Gaussian distribution
    parmest.graphics.pairwise_plot((theta, cov, 100), theta_star=theta, alpha=0.8, 
                                   distributions=['MVN'], title='Theta estimates within 80% confidence region')
    
    # Assert statements compare parameter estimation (theta) to an expected value 
    relative_error = abs(theta['asymptote'] - 19.1426)/19.1426
    assert relative_error < 0.01
    relative_error = abs(theta['rate_constant'] - 0.5311)/0.5311
    assert relative_error < 0.01
    
    return theta, cov

def bootstrap_resampling(pest):
    # Parameter estimation with bootstrap resampling
    obj, theta = pest.theta_est()
    bootstrap_theta = pest.theta_est_bootstrap(50, seed=4581)
    
    # Plot results
    parmest.graphics.pairwise_plot(bootstrap_theta, title='Bootstrap theta')
    parmest.graphics.pairwise_plot(bootstrap_theta, theta, 0.8, ['MVN', 'KDE', 'Rect'], 
                          title='Bootstrap theta with confidence regions')

    return bootstrap_theta

def likelihood_ratio_test(pest):
    # Find the objective value at each theta estimate
    asym = np.arange(10, 30, 2)
    rate = np.arange(0, 1.5, 0.1)
    theta_vals = pd.DataFrame(list(product(asym, rate)), columns=theta_names)
    obj, theta = pest.theta_est()
    obj_at_theta = pest.objective_at_theta(theta_vals)
    
    # Run the likelihood ratio test
    LR = pest.likelihood_ratio_test(obj_at_theta, obj, [0.8, 0.85, 0.9, 0.95])
    
    # Plot results
    parmest.graphics.pairwise_plot(LR, theta, 0.8, 
                          title='LR results within 80% confidence region')
    
    return obj_at_theta, LR

    
if __name__ == "__main__":
    
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
    
    # Create an instance of the parmest estimator
    pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE)
    
    # Parameter estimation and covariance
    theta, cov = parameter_estimation(pest)
    print(theta)
    print(cov)
    
    # Parameter estimation with bootstrap resampling
    bootstrap_theta = bootstrap_resampling(pest)
    print(bootstrap_theta.head())
    
    # Likelihood ratio test
    obj_at_theta, LR = likelihood_ratio_test(pest)
    print(LR.head())
    
    