#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for 
model parameter uncertainty using nonlinear confidence regions. AIChE Journal, 
47(8), 1794-1804.
"""
import pandas as pd
from pyomo.environ import ConcreteModel, Var, Expression, Objective, SolverFactory, exp, minimize

def rooney_biegler_model(data):
    
    model =  ConcreteModel()

    model.asymptote =  Var(initialize = 15)
    model.rate_constant =  Var(initialize = 0.5)
    
    def response_rule(m, h):
        expr = m.asymptote * (1 -  exp(-m.rate_constant * h))
        return expr
    model.response_function =  Expression(data.hour, rule = response_rule)
    
    def SSE_rule(m):
        return sum((data.y[i] - m.response_function[data.hour[i]])**2 for i in data.index)
    model.SSE =  Objective(rule = SSE_rule, sense= minimize)
    
    return model

if __name__ == '__main__':
    
    data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],[4,16.0],[5,15.6],[6,19.8]],
                        columns=['hour', 'y'])
    
    model = rooney_biegler_model(data)
    solver =  SolverFactory('ipopt')
    solver.solve(model)
    print('asymptote = ', model.asymptote())
    print('rate constant = ', model.rate_constant())
