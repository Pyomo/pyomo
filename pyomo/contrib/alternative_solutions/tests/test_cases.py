#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import random
from itertools import product

import numpy as np

import pyomo.environ as pe

def get_2d_diamond_problem(discrete_x=False, discrete_y=False):
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.Integers if discrete_x else pe.Reals)
    m.y = pe.Var(within=pe.Integers if discrete_y else pe.Reals)
    
    m.o = pe.Objective(expr = m.x + m.y, sense=pe.maximize)
    
    m.c1 = pe.Constraint(expr= -4/5 * m.x - 4 <= m.y)
    m.c2 = pe.Constraint(expr=  5/9 * m.x - 5 <= m.y)
    m.c3 = pe.Constraint(expr=  2/9 * m.x + 2 >= m.y)
    m.c4 = pe.Constraint(expr= -1/2 * m.x + 3 >= m.y)
    #m.c5 = pe.Constraint(expr= 2.30769230769231 >= m.y)

    m.extreme_points = {(0.737704918, -4.590163934),
                        (-5.869565217, 0.695652174),
                        (1.384615385, 2.307692308),
                        (7.578947368, -0.789473684)}

    m.continuous_bounds = pe.ComponentMap()
    m.continuous_bounds[m.x] = (-5.869565217, 7.578947368)
    m.continuous_bounds[m.y] = (-4.590163934, 2.307692308)

    return m

def get_aos_test_knapsack(var_max, weights, values, capacity_fraction):
    assert len(weights) == len(values), \
        'weights and values must be the same length.'
    assert 0 <= capacity_fraction and capacity_fraction <= 1, \
            'capacity_fraction must be between 0 and 1.'
    
    num_vars = len(weights)
    capacity = sum(weights) * var_max * capacity_fraction
    
    m = pe.ConcreteModel()
    m.i = pe.RangeSet(0,num_vars-1)
    m.x = pe.Var(m.i, within=pe.NonNegativeIntegers, bounds=(0,var_max))

    m.o = pe.Objective(expr=sum(values[i]*m.x[i] for i in m.i), 
                       sense=pe.maximize)

    m.c = pe.Constraint(expr=sum(weights[i]*m.x[i] for i in m.i) <= capacity)
    
    var_domain = var_values = range(var_max+1)
    all_combos = product(var_domain, repeat=num_vars)
    
    feasible_sols = []
    for sol in all_combos:
        if np.dot(sol, weights) <= capacity:
            feasible_sols.append((sol, np.dot(sol, values)))
    sorted(feasible_sols, key=lambda sol: sol[1], reverse=False)
    print(feasible_sols)
    return m



# from pyomo.contrib.alternative_solutions.obbt import obbt_analysis

# def get_random_knapsack_model(num_x_vars, num_y_vars, budget_pct, seed=1000):
#     random.seed(seed)
    
#     W = budget_pct * (num_x_vars + num_y_vars) / 2
    
    
#     model = pe.ConcreteModel()
    
#     model.X_INDEX = pe.RangeSet(1,num_x_vars)
#     model.Y_INDEX = pe.RangeSet(1,num_y_vars)
    
#     model.wu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.vu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.x = pe.Var(model.X_INDEX, within=pe.NonNegativeIntegers)
    
#     model.b = pe.Block()
#     model.b.wl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.b.vl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
#     model.b.y = pe.Var(model.Y_INDEX, within=pe.NonNegativeReals)
    
#     model.o = pe.Objective(expr=sum(model.vu[i]*model.x[i] for i in model.X_INDEX) + \
#                            sum(model.b.vl[i]*model.b.y[i] for i in model.Y_INDEX), sense=pe.maximize)
#     model.c = pe.Constraint(expr=sum(model.wu[i]*model.x[i] for i in model.X_INDEX) + \
#                             sum(model.b.wl[i]*model.b.y[i] for i in model.Y_INDEX)<= W)
        
#     return model

# model = get_random_knapsack_model(4, 4, 0.2)
# result = obbt_analysis(model, variables='all', rel_opt_gap=None, 
#                                   abs_gap=None, already_solved=False, 
#                                   solver='gurobi', solver_options={}, 
#                                   use_persistent_solver = False, tee=True,
#                                   refine_bounds=False)