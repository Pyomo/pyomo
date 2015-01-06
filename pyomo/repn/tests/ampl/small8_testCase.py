#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly modifies product expressions 
#          with only constant terms in the denominator (that
#          are involved in linear expressions).
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import *

model = AbstractModel()

model.a = Param(initialize=2.0)

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.z = Var(within=NonNegativeReals,bounds=(7,None))

def obj_rule(model):
    return model.z + model.x*model.x + model.y
model.obj = Objective(rule=obj_rule,sense=minimize)

def constr_rule(model):
    return (model.a,model.y*model.y,None)
model.constr = Constraint(rule=constr_rule)

def constr2_rule(model):
    return model.x/model.a >= model.y
model.constr2 = Constraint(rule=constr2_rule)

def constr3_rule(model):
    return model.z <= model.y + model.a
model.constr3 = Constraint(rule=constr3_rule)

