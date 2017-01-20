#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *

M = ConcreteModel()
M.x = Var([1,2,3], within=Boolean)

M.o = Objective(expr=summation(M.x))
M.c1 = Constraint(expr=4*M.x[1]+M.x[2] >= 1)
M.c2 = Constraint(expr=M.x[2]+4*M.x[3] >= 1)

model=M
