import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as grph

def rooney_biegler_model(data):
    
    model = pyo.ConcreteModel()

    model.asymptote = pyo.Var(initialize = 15)
    model.rate_constant = pyo.Var(initialize = 0.5)
    
    def response_rule(m, h):
        expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
        return expr
    model.response_function = pyo.Expression(data.hour, rule = response_rule)
    
    def SSE_rule(m):
        return sum((data.y[i] - m.response_function[data.hour[i]])**2 for i in data.index)
    model.SSE = pyo.Objective(rule = SSE_rule, sense=pyo.minimize)
    
    return model

# Data, which would be loaded from an excel or text file and could have 
# differnet column names than were used in the model
# 0 indexed Dataframe
data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],[4,16.0],[5,15.6],[6,19.8]],
                    columns=['time', 'sensor12'])
# Rename columns to match the model
data.rename(columns={'time':'hour','sensor12':'y'}, inplace=True)

# Use the model directly
model = rooney_biegler_model(data)
model.display()
solver = pyo.SolverFactory('ipopt')
solver.solve(model)
print('asymptote = ', model.asymptote())
print('rate constant = ', model.rate_constant())

# Possible new structure for parmest
# This modifies the pyomo model within parmest and using a callback in parmest
# parmest.Estimator just wraps ParmEstimator to test out the new structure
thetalist = ['asymptote', 'rate_constant']
def second_stage_cost(model, data): # Can different from the current model objective
    return sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
pest = parmest.Estimator(rooney_biegler_model, data, thetalist, second_stage_cost)
objval, thetavals = pest.theta_est()
print ("objective value=",str(objval))
print ("theta-star=",str(thetavals))

# Bootstrap
alpha = 0.8
num_bootstraps = 10
bootstrap_theta = pest.bootstrap(num_bootstraps)
print ("Bootstrap:")
print(bootstrap_theta)
grph.pairwise_plot(bootstrap_theta, filename="RB.png")
grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, alpha, 
                                 filename="RB_boot.png")

# LR
search_ranges = {}
search_ranges["asymptote"] = np.arange(10, 30, 2) # np.arange(10, 30, 0.01)
search_ranges["rate_constant"] = np.arange(0, 1.5, 0.1) # np.arange(0, 1.5, 0.005)
SSE = pest.likelihood_ratio(search_ranges=search_ranges)
print ("Likelihood Ratio:")
print(SSE)
grph.pairwise_likelihood_ratio_plot(SSE, objval, alpha, data.shape[0], 
                                        filename="RB_LR.png")

"""
# Multiple data points per measurment???
data = pd.DataFrame(data=[[1,8.3,None,None],[2,10.3,None,None],[3,19.0,18.5,19.2],
                                      [4,16.0,16.1,None],[5,15.6,None,15.2],[6,19.8,None,None]],
                                      columns=['time', 'y1','y2','y3'])
column_name_map = {'time': 'hour', 'y1': 'y', 'y2': 'y', 'y3':'y'}
data.rename(columns=column_name_map, inplace=True)
print(data['y']) # returns three columns

def SecondStageCost2(model, data): # Must be formatted for a single experiment
    return sum((ys - model.response_function[data.hour[i]])**2 for ys in data.loc[i,'y'] for i in data.index)

pest = parmest.Estimator(rooney_biegler_model, data, thetalist, SecondStageCost2)
objval, thetavals = pest.theta_est()
"""