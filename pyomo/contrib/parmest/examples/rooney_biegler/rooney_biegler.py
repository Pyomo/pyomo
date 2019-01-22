"""
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for 
model parameter uncertainty using nonlinear confidence regions. AIChE Journal, 
47(8), 1794-1804.
"""
import pandas as pd
import pyomo.environ as pyo

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

if __name__ == '__main__':
    
    data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],[4,16.0],[5,15.6],[6,19.8]],
                        columns=['hour', 'y'])
    
    model = rooney_biegler_model(data)
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)
    print('asymptote = ', model.asymptote())
    print('rate constant = ', model.rate_constant())
