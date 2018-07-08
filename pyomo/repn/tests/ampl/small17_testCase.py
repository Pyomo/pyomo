#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Author:  William Hart
# Purpose: For regression testing to ensure that the Pyomo
#          writers generate files in a consistent manner when
#          using index sets with strings and integers.
#

from pyomo.environ import *

model = ConcreteModel()

model.A = Set(initialize=[2, '1', 0], within=Any)
def x_bounds(model, i):
    if i == 2:
        return (2, None)
    if i == 0:
        return (0, None)
    return (1, None)
model.x = Var(model.A, initialize=1.0, bounds=x_bounds)

model.OBJ = Objective(expr=model.x[2]+model.x['1'])

def c_rule(model, i):
    if i == 2:
        return model.x[2] + model.x[0] == 1
    if i == '1':
        return model.x['1'] + model.x[0] == 1
    return Constraint.Skip
model.c = Constraint(model.A, rule=c_rule)
