#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Knapsack Problem
#

import pyomo.environ as pyo

v = {'hammer': 8, 'wrench': 3, 'screwdriver': 6, 'towel': 11}
w = {'hammer': 5, 'wrench': 7, 'screwdriver': 4, 'towel': 3}

limit = 14

M = pyo.ConcreteModel()

M.ITEMS = pyo.Set(initialize=v.keys())

M.x = pyo.Var(M.ITEMS, within=pyo.Binary)

M.value = pyo.Objective(expr=sum(v[i] * M.x[i] for i in M.ITEMS), sense=pyo.maximize)

M.weight = pyo.Constraint(expr=sum(w[i] * M.x[i] for i in M.ITEMS) <= limit)
