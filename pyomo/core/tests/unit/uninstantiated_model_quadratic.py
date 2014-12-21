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

model.x = Var()

def objective_rule ( M ):
    return M.x * M.x    # should fail "gracefully"

model.objective = Objective(rule=objective_rule, sense=minimize)
