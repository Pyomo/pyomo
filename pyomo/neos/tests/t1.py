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
M.x = Var(bounds=(0,1))
M.o = Objective(expr=2*M.x, sense=maximize)
M.c = Constraint(expr=M.x <= 0.5)

model = M

