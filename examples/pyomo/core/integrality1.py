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

import pyomo.environ as pyo

M = pyo.ConcreteModel()
M.x1 = pyo.Var(within=pyo.Boolean)
M.x2 = pyo.Var(within=pyo.Boolean)
M.x3 = pyo.Var(within=pyo.Boolean)

M.o = pyo.Objective(expr=M.x1 + M.x2 + M.x3)
M.c1 = pyo.Constraint(expr=4 * M.x1 + M.x2 >= 1)
M.c2 = pyo.Constraint(expr=M.x2 + 4 * M.x3 >= 1)

model = M
