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

model.indices = Set(initialize=[1,2])

model.p = Param(model.indices)

model.x = Var(model.indices)

def objective_rule ( M ):
    return sum([M.p[i] * M.x[i] for i in model.indices])

model.objective = Objective(rule=objective_rule, sense=minimize)
