#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import AbstractModel, RangeSet, Var, Objective, sum_product

model = AbstractModel()

model.A = RangeSet(1,4)

model.x = Var(model.A)

def obj_rule(model):
    return sum_product(model.x)
model.obj = Objective(rule=obj_rule)
