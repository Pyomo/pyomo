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

import pyomo.environ as pe

def get_continuous_prob_1(discrete_x=False, discrete_y=False):
    m = pe.ConcreteModel()
    m.x = pe.Var(within=pe.Integers if discrete_x else pe.Reals)
    m.y = pe.Var(within=pe.Integers if discrete_y else pe.Reals)
    
    m.o = pe.Objective(expr = m.x + m.y, sense=pe.maximize)
    
    m.c1 = pe.Constraint(expr= -4/5 * m.x - 4 <= m.y)
    m.c2 = pe.Constraint(expr=  5/9 * m.x - 5 <= m.y)
    m.c3 = pe.Constraint(expr=  2/9 * m.x + 2 >= m.y)
    m.c4 = pe.Constraint(expr= -1/2 * m.x + 3 >= m.y)
    m.c5 = pe.Constraint(expr= 2.30769230769231 >= m.y)

    return m

def knapsack(N):
    random.seed(1000)

    N = N
    W = N/10.0


    model = pe.ConcreteModel()

    model.INDEX = pe.RangeSet(1,N)

    model.w = pe.Param(model.INDEX, initialize=lambda model, i : random.uniform(0.0,1.0), within=pe.Reals)

    model.v = pe.Param(model.INDEX, initialize=lambda model, i : random.uniform(0.0,1.0), within=pe.Reals)

    model.x = pe.Var(model.INDEX, within=pe.Binary)

    model.o = pe.Objective(expr=sum(model.v[i]*model.x[i] for i in model.INDEX), sense=pe.maximize)

    model.c = pe.Constraint(expr=sum(model.w[i]*model.x[i] for i in model.INDEX) <= W)

    return model