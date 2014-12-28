#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *

model = ConcreteModel()

model.x1 = Var()
model.x2 = Var(bounds=(-1,1))
model.x3 = Var(bounds=(1,2))

def obj_rule(m):
    return m.x1**2 + (m.x2*m.x3)**4 + \
          m.x1*m.x3 + \
          m.x2*sin(m.x1+m.x3) + m.x2
model.obj = Objective(rule=obj_rule)

