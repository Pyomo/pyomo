#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import AbstractModel, Var, Objective, minimize

model = AbstractModel()

model.x = Var()

def objective_rule ( M ):
    return M.x * M.x    # should fail "gracefully"

model.objective = Objective(rule=objective_rule, sense=minimize)
