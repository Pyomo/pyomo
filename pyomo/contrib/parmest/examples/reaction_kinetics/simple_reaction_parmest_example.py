import pandas as pd
from pandas import DataFrame
from os import path

from pyomo.environ import (ConcreteModel, Param, Var, PositiveReals, Objective,
                           Constraint, RangeSet, Expression, minimize, exp, value)

#from idaes.core.util import get_default_solver
import pyomo.contrib.parmest.parmest as parmest

# =======================================================================
data = [{'experiment': 1, 'x1': 0.1, 'x2': 100, 'y': 0.98},
        {'experiment': 2, 'x1': 0.2, 'x2': 100, 'y': 0.983},
        {'experiment': 3, 'x1': 0.3, 'x2': 100, 'y': 0.955},
        {'experiment': 4, 'x1': 0.4, 'x2': 100, 'y': 0.979},
        {'experiment': 5, 'x1': 0.5, 'x2': 100, 'y': 0.993},
        {'experiment': 6, 'x1': 0.05, 'x2': 200, 'y': 0.626},
        {'experiment': 7, 'x1': 0.1, 'x2': 200, 'y': 0.544},
        {'experiment': 8, 'x1': 0.15, 'x2': 200, 'y': 0.455},
        {'experiment': 9, 'x1': 0.2, 'x2': 200, 'y': 0.225},
        {'experiment': 10, 'x1': 0.25, 'x2': 200, 'y': 0.167},
        {'experiment': 11, 'x1': 0.02, 'x2': 300, 'y': 0.566},
        {'experiment': 12, 'x1': 0.04, 'x2': 300, 'y': 0.317},
        {'experiment': 13, 'x1': 0.06, 'x2': 300, 'y': 0.034},
        {'experiment': 14, 'x1': 0.08, 'x2': 300, 'y': 0.016},
        {'experiment': 15, 'x1': 0.1, 'x2': 300, 'y': 0.006}]

# =======================================================================

def simple_reaction_model(data):

    # Create the concrete model
    model = ConcreteModel()

    model.x1 = Param(initialize=float(data['x1']))
    model.x2 = Param(initialize=float(data['x2']))

    # Rate constants
    model.rxn = RangeSet(2)
    initial_guess = {1: 750, 2: 1200}
    model.k = Var(model.rxn, initialize=initial_guess, within=PositiveReals)

    # reaction product
    model.y = Expression(expr=exp(-model.k[1] *
                                  model.x1 * exp(-model.k[2] / model.x2)))
                                  
    # fix the rate constants
    model.k.fix()

    # linked variables and constraints
    #model.k1 = Var(initialize=750, within=PositiveReals)
    #model.k2 = Var(initialize=1200, within=PositiveReals)
    #model.eq_L1 = Constraint(expr = model.k1 == model.k[1])
    #model.eq_L2 = Constraint(expr = model.k2 == model.k[2])
    #===================================================================
    # Stage-specific cost computations
    def ComputeFirstStageCost_rule(model):
        return 0
    model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

    def AllMeasurements(m):
        return (float(data['y']) - m.y) ** 2
    model.SecondStageCost = Expression(rule=AllMeasurements)

    def total_cost_rule(m):
        return m.FirstStageCost + m.SecondStageCost
    model.Total_Cost_Objective = Objective(rule=total_cost_rule,
                                           sense=minimize)

    return model

if __name__ == "__main__":

    # =======================================================================
    # Parameter estimation without covariance estimate
    #solver = get_default_solver
    theta_names = ['k[1]']
    pest = parmest.Estimator(simple_reaction_model, data, theta_names)
    obj, theta = pest.theta_est()
    print(obj)
    print(theta)
    #=======================================================================
    # Parameter estimation covariance estimate
    
    obj, theta, cov = pest.theta_est(calc_cov=True)
    print(obj)
    print(theta)
    print(cov)
    
    #=======================================================================