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
model.S = RangeSet(10)
model.X = Var(model.S)

def C_rule(i, m):
    return m.X[i] >= 10.0
model.C = Constraint(rule=C_rule)
