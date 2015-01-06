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
#          NL writer properly reclassifies nonlinear expressions
#          as linear or trivial when fixing variables or params
#          cause such a situation.
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import *

model = ConcreteModel()

model.x = Var()
model.y = Var()
model.z = Var()
model.q = Param(initialize=0.0)
model.p = Param(initialize=0.0,mutable=True)

model.obj = Objective( expr=model.x*model.y +\
                            model.z*model.y +\
                            model.q*model.y +\
                            model.y*model.y*model.q +\
                            model.p*model.y +\
                            model.y*model.y*model.p +\
                            model.y*model.y*model.z +\
                            model.z*(model.y**2))

model.con1 = Constraint(expr=model.x*model.y == 0)
model.con2 = Constraint(expr=model.z*model.y + model.y == 0)
model.con3 = Constraint(expr=model.q*(model.y**2) + model.y == 0)
model.con4 = Constraint(expr=model.q*model.y*model.x + model.y == 0)
model.con5 = Constraint(expr=model.p*(model.y**2) + model.y == 0)
model.con6 = Constraint(expr=model.p*model.y*model.x + model.y == 0)
model.con7 = Constraint(expr=model.z*(model.y**2) + model.y == 0)
model.con8 = Constraint(expr=model.z*model.y*model.x + model.y == 0)
# Pyomo differs from AMPL in these cases that involve immutable params (q).
# These never actually become constraints in Pyomo, and for good reason.
model.con9 = Constraint(expr=model.z*model.y == 0)
model.con10 = Constraint(
    rule= simple_constraint_rule(model.q*(model.y**2) == 0) )
model.con11 = Constraint(
    rule= simple_constraint_rule(model.q*model.y*model.x == 0) )
model.con12 = Constraint(expr=model.p*(model.y**2) == 0)
model.con13 = Constraint(expr=model.p*model.y*model.x == 0)
model.con14 = Constraint(expr=model.z*(model.y**2) == 0)
model.con15 = Constraint(expr=model.z*model.y*model.x == 0)
model.con16 = Constraint(
    rule= simple_constraint_rule(model.q*model.y == 0) )
model.con17 = Constraint(expr=model.p*model.y == 0)

###### Add some constraint which we deactivate just
###### to make sure this is working properly
model.con1D = Constraint(expr=model.x*model.y == 0)
model.con1D_indexeda = Constraint([1,2],rule=lambda model,i: model.x*model.y == 0)
model.con1D_indexedb = Constraint([1,2],rule=lambda model,i: model.x*model.y == 0)
model.con1D.deactivate()
model.con1D_indexeda.deactivate()
model.con1D_indexedb[1].deactivate()
model.con1D_indexedb[2].deactivate()
#####


model.x = 1.0
model.x.fixed = True
model.z = 0.0
model.z.fixed = True
