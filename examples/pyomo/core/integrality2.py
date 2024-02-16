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

from pyomo.environ import *

M = ConcreteModel()
M.x = Var([1, 2, 3], within=Boolean)

M.o = Objective(expr=sum_product(M.x))
M.c1 = Constraint(expr=4 * M.x[1] + M.x[2] >= 1)
M.c2 = Constraint(expr=M.x[2] + 4 * M.x[3] >= 1)

model = M
