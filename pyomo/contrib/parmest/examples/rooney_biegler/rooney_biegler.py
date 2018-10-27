# This file contains the Rooney Biegler model and callback

import pandas as pd
import pyomo.environ as pyo

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

def instance_creation_callback(experiment_number = None, cb_data = None):
    
    model = generate_model(cb_data.loc[experiment_number,:])
    
    return model


if __name__ == '__main__':
    
    experiment_data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                      [4,16.0],[5,15.6],[6,19.8]],
                                      columns=['hour', 'y'])
    
    model = instance_creation_callback(2, experiment_data)
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)
    model.display()
    model.pprint()
    print('asymptote = ', model.asymptote())
    print('rate constant = ', model.rate_constant())
