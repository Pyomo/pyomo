#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# a slightly less brain-dead separable parabolic function of three variables, whose
# minimum is obviously at x[1]=x[2]=x[3]=0.

from pyomo.core import *

model = AbstractModel()

def indices_rule(model):
    return xrange(1,4)
model.indices = Set(initialize=indices_rule, within=PositiveIntegers)

model.x = Var(model.indices, within=Reals)

def bound_x_rule(model, i):
    return (-10, model.x[i], 10)
model.bound_x = Constraint(model.indices, rule=bound_x_rule)

def objective_rule(model):
    return sum([model.x[i] * model.x[i] for i in model.indices])
model.objective = Objective(rule=objective_rule, sense=minimize)
