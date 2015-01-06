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
model.X = Var()

def c_rule(m):
    try:
        return model.X >= 10.0
    except Exception:
        return model.X >= 20.0
model.C = Constraint(rule=c_rule)
