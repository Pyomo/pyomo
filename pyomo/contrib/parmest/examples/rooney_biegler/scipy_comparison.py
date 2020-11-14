#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

'''
This file solves the Rooney and Bielger example with scipy. The main purpose
is to compare the parameter estimation results and covariance matrix approximation.
'''

import numpy as np
import scipy.optimize as optimize
import pandas as pd

def model(theta, t):
    '''
    Model to be fitted y = model(theta, t)
    Arguments:
        theta: vector of fitted parameters
        t: independent variable [hours]
        
    Returns:
        y: model predictions [need to check paper for units]
    '''
    asymptote = theta[0]
    rate_constant = theta[1]
    
    return asymptote * (1 - np.exp(-rate_constant * t))

def residual(theta, t, y):
    '''
    Calculate residuals
    Arguments:
        theta: vector of fitted parameters
        t: independent variable [hours]
        y: dependent variable [?]
    '''
    return y - model(theta, t)

# define data
# we are using a pandas dataframe for parity with the pyomo example
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],[4,16.0],[5,15.6],[7,19.8]],
                        columns=['hour', 'y'])
t = data['hour'].to_numpy()
y = data['y'].to_numpy()

# define initial guess
theta_guess = np.array([15, 0.5])

## solve with optimize.least_squares
print("Regress with OPTIMIZE.LEAST_SQUARES")

# solve
sol = optimize.least_squares(residual, theta_guess,method='trf',args=(t,y),verbose=2)
theta_hat = sol.x

print("\n")
print("asymptote =", theta_hat[0])
print("rate_constant =", theta_hat[1])

# calculate residuals
r = residual(theta_hat, t, y)

# calculate variance of the residuals
# -2 because there are 2 fitted parameters
sigre = np.matmul(r.T, r / (len(y) - 2))
print("sigre = ", sigre)

# approximate covariance
# Need to divide by 2 because optimize.least_squares scaled the objective by 1/2
cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))

print("\ncovariance=\n", cov)

## solve with optimize.curve_fit
print("Regress with OPTIMIZE.CURVE_FIT")

def model2(t, asymptote, rate_constant):
    return asymptote * (1 - np.exp(-rate_constant * t))

theta_hat2, cov2 = optimize.curve_fit(model2, t, y, p0=theta_guess)

print("\n")
print("asymptote =", theta_hat2[0])
print("rate_constant =", theta_hat2[1])

print("\ncovariance=\n", cov2)


## covariance matrix in Rooney and Biegler (2001)
cov_paper = np.array([[6.22864, -0.4322], [-0.4322, 0.04124]])
print("\ncovariance from paper =\n", cov_paper)

'''
These scipy results differ in the 3rd decimal place from the paper. It is possible
the paper used an alternative finite difference approximation for the Jacobian.
'''
