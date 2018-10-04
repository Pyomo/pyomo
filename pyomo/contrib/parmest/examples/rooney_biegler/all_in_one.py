# This file defines the Rooney Biegler model, data, callback, and driver

import re
import pandas as pd
import numpy as np
import pyomo.environ as pyo

data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],[4,16.0],[5,15.6],[6,19.8]],
                    columns=['hour', 'y'])

def generate_model(data):
    
    h = int(data['hour'])
    y = float(data['y'])
    
    model = pyo.ConcreteModel()
    model.asymptote = pyo.Var(initialize=15)
    model.rate_constant = pyo.Var(initialize=0.5)

    def response_rule(m):
        expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
        return expr

    model.response_function = pyo.Expression(rule=response_rule)

    def SSE_rule(m):
        return (y - m.response_function) ** 2

    def ComputeFirstStageCost_rule(m):
        return 0

    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)
    model.SecondStageCost = pyo.Expression(rule=SSE_rule)

    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost

    # This objective is not needed by parmest
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, 
                                               sense=pyo.minimize)
    
    return model

def pysp_instance_creation_callback(scenario_tree_model,
                                    scenario_name,
                                    node_names):
    
    if isinstance(scenario_name, str): # if scenario is a str, get the trailing integer
        index = int(re.compile(r'(\d+)$').search(scenario_name).group(1))
    elif isinstance(scenario_name, int):
        index = scenario_name
    else:
        return
    
    model = generate_model(data.loc[index,:])
    
    return model


if __name__ == '__main__':
    # Very simple, with just theta estimation and bootstrap
    # Not done in parallel.
    import pyomo.contrib.parmest.parmest as parmest
    import pyomo.contrib.parmest.graphics as grph

    # prepare for the parmest object construction
    num_samples = 6
    sample_list = list(range(num_samples)) 
    thetalist = ['asymptote', 'rate_constant']

    np.random.seed(1134)

    # Generate parmest object using the callback function itself,
    #   rather than a module and a function name
    pest = parmest.ParmEstimator(None, pysp_instance_creation_callback,
                                 "SecondStageCost", sample_list, thetalist)

    ### Parameter estimation with entire data set
    objval, thetavals = pest.theta_est()

    print ("objective value=",str(objval))
    print ("theta-star=",str(thetavals))

    ### Parameter estimation with bootstrap
    num_bootstraps = 10
    bootstrap_theta = pest.bootstrap(num_bootstraps)
    print ("Bootstrap:")
    print(bootstrap_theta)
    grph.pairwise_bootstrap_plot(bootstrap_theta, thetavals, 0.8, 
                                 filename="RB_boot.png")
