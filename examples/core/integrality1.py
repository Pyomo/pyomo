#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *

M = ConcreteModel()
M.x1 = Var(within=Boolean)
M.x2 = Var(within=Boolean)
M.x3 = Var(within=Boolean)

M.o = Objective(expr=M.x1+M.x2+M.x3)
M.c1 = Constraint(expr=4*M.x1+M.x2 >= 1)
M.c2 = Constraint(expr=M.x2+4*M.x3 >= 1)

model=M
