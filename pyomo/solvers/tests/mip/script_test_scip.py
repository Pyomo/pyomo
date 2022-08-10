#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:33:42 2022

@author: pmlpm
"""

import pyomo.environ as pyo

import random

#******************************************************************************
#******************************************************************************

# carry out optimisations

def optimise(problem: pyo.ConcreteModel,
             solver_timelimit,
             solver_rel_mip_gap,
             solver_abs_mip_gap,
             print_solver_output: bool = True):
    
    # config
    
    options_dict_format = {
        'limits/time':solver_timelimit,
        'limits/gap':solver_rel_mip_gap,
        'limits/absgap':solver_abs_mip_gap
        }
    
    opt = pyo.SolverFactory('scip')
    
    for key, value in options_dict_format.items():
        
        opt.options[key] = value
    
    # solve
    
    results = opt.solve(
        problem, 
        tee=print_solver_output
        )
    
    # return
    
    return results

#******************************************************************************
#******************************************************************************

def problem_lp_optimal():
    
    model = pyo.ConcreteModel('lp_optimal')
    
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
        
    return model

def problem_lp_infeasible():
    
    model = pyo.ConcreteModel('lp_infeasible')
    
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] <= -1)
        
    return model

def problem_lp_unbounded():
    
    model = pyo.ConcreteModel('lp_unbounded')
    
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2], 
                              sense=pyo.maximize)
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
        
    return model

def problem_milp_optimal():
    
    model = pyo.ConcreteModel('milp_optimal')
    
    model.x = pyo.Var([1,2], domain=pyo.Binary)
    
    model.OBJ = pyo.Objective(expr = 2.15*model.x[1] + 3.8*model.x[2])
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
    
    return model

def problem_milp_infeasible():
    
    model = pyo.ConcreteModel('milp_infeasible')
        
    model.x = pyo.Var([1,2], domain=pyo.Binary)
    
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] <= -1)
        
    return model

def problem_milp_unbounded():
    
    model = pyo.ConcreteModel('milp_unbounded')
    
    model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    
    model.y = pyo.Var(domain=pyo.Binary)
    
    model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2] + model.y, 
                              sense=pyo.maximize)
    
    model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
        
    return model

def problem_milp_feasible():
    
    model = pyo.ConcreteModel('milp_feasible')
    
    random.seed(6254)
    
    # a knapsack-type problem
    
    number_binary_variables = 100
    
    model.Y = pyo.RangeSet(number_binary_variables)
    
    model.y = pyo.Var(model.Y,
                      domain=pyo.Binary)
    
    model.OBJ = pyo.Objective(
        expr = sum(model.y[j]*random.random()
                   for j in model.Y), 
        sense=pyo.maximize
        )
    
    model.Constraint1 = pyo.Constraint(
        expr = sum(model.y[j]*random.random()
                   for j in model.Y) <= round(number_binary_variables/5)
        )
    
    def rule_c1(m, i):
        return (
            sum(model.y[j]*(random.random()-0.5)
                for j in model.Y
                if j != i
                if random.randint(0,1)
                ) <= round(number_binary_variables/5)*model.y[i]
            )
    model.constr_c1 = pyo.Constraint(
        model.Y,
        rule=rule_c1)
    
    return model

#******************************************************************************
#******************************************************************************

# list of problems

list_concrete_models = [
    problem_lp_unbounded(),
    problem_lp_infeasible(),
    problem_lp_optimal(),
    problem_milp_unbounded(),
    problem_milp_infeasible(),
    problem_milp_optimal(),
    problem_milp_feasible()
    ]

#******************************************************************************
#******************************************************************************

# solver settings

solver_timelimit = 12

solver_abs_mip_gap = 0

solver_rel_mip_gap = 1e-6

#******************************************************************************
#******************************************************************************

list_problem_results = []

for problem in list_concrete_models:
    
    print('******************************')
    print('******************************')
    
    print(problem.name)
        
    print('******************************')
    print('******************************')
    
    results = optimise(problem, 
                       solver_timelimit,
                       solver_rel_mip_gap,
                       solver_abs_mip_gap,
                       print_solver_output=True)
    
    print(results)
    
    list_problem_results.append(results)

#******************************************************************************
#******************************************************************************