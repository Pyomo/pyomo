#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.environ import *

model = AbstractModel()
model.S = RangeSet(10)
model.X = Var(model.S)

def c_rule(m, i):
    if sum(m.X[i] for i in m.S) <= 10.0:
        pass
    if sum(value(m.X[i]) for i in m.S) <= 10.0:
        pass
    return sum(m.X[i] for i in m.S) <= 10.0
model.C = Constraint(rule=c_rule)
