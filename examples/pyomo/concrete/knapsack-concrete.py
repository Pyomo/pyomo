#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Knapsack Problem
#

from pyomo.environ import *

v = {'hammer': 8, 'wrench': 3, 'screwdriver': 6, 'towel': 11}
w = {'hammer': 5, 'wrench': 7, 'screwdriver': 4, 'towel': 3}

limit = 14

M = ConcreteModel()

M.ITEMS = Set(initialize=v.keys())

M.x = Var(M.ITEMS, within=Binary)

M.value = Objective(expr=sum(v[i] * M.x[i] for i in M.ITEMS), sense=maximize)

M.weight = Constraint(expr=sum(w[i] * M.x[i] for i in M.ITEMS) <= limit)
