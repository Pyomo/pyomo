# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:13:06 2022

@author: jlgearh
"""

import random

import pyomo.environ as pe

import pyomo.contrib.alternative_solutions.balas as bls


def get_random_knapsack_model(num_x_vars, num_y_vars, budget_pct, seed=1000):
    random.seed(seed)
    
    W = budget_pct * (num_x_vars + num_y_vars) / 2
    
    
    model = pe.ConcreteModel()
    
    model.X_INDEX = pe.RangeSet(1,num_x_vars)
    model.Y_INDEX = pe.RangeSet(1,num_y_vars)
    
    model.wu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
    model.vu = pe.Param(model.X_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
    model.x = pe.Var(model.X_INDEX, within=pe.Binary)
    
    model.b = pe.Block()
    model.b.wl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
    model.b.vl = pe.Param(model.Y_INDEX, initialize=lambda model, i : round(random.uniform(0.0,1.0), 2), within=pe.Reals)
    model.b.y = pe.Var(model.Y_INDEX, within=pe.Binary)
    
    model.o = pe.Objective(expr=sum(model.vu[i]*model.x[i] for i in model.X_INDEX) + \
                           sum(model.b.vl[i]*model.b.y[i] for i in model.Y_INDEX), sense=pe.maximize)
    model.c = pe.Constraint(expr=sum(model.wu[i]*model.x[i] for i in model.X_INDEX) + \
                            sum(model.b.wl[i]*model.b.y[i] for i in model.Y_INDEX)<= W)
        
    return model

model = get_random_knapsack_model(4, 4, 0.2)

alternative_solutions = bls.enumerate_binary_solutions(model,
                                                       search_mode='hamming',
                                                       max_solutions = 99)