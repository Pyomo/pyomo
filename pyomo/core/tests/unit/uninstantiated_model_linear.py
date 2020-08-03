#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import AbstractModel, Set, Param, Var, Objective, minimize

model = AbstractModel()

model.indices = Set(initialize=[1,2])

model.p = Param(model.indices)

model.x = Var(model.indices)

def objective_rule ( M ):
    return sum([M.p[i] * M.x[i] for i in model.indices])

model.objective = Objective(rule=objective_rule, sense=minimize)
