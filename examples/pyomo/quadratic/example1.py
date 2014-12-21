#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# a brain-dead parabolic function of a single variable, whose minimum is
# obviously at x=0. the entry-level quadratic test-case. the lack of
# constraints could (but shouldn't in a perfect world) cause issues for
# certain solvers.

from pyomo.core import *

model = AbstractModel()

model.x = Var(bounds=(-10,10), within=Reals)

def objective_rule(model):
    return model.x * model.x
model.objective = Objective(rule=objective_rule, sense=minimize)
