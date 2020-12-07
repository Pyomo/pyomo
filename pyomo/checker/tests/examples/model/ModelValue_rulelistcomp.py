#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
