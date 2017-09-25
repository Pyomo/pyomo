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
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reclassifies nonlinear expressions
#          as linear or trivial when fixing variables or params
#          cause such a situation.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import *

model = ConcreteModel()

model.x = Var()
model.y = Var(initialize=0.0)
model.z = Var()
model.p = Param(initialize=0.0,mutable=True)
model.q = Param(initialize=0.0,mutable=False)

model.y.fixed = True

model.obj = Objective( expr=model.x )

model.con1 = Constraint(expr= model.x*model.y*model.z + model.x == 1.0)
model.con2 = Constraint(expr= model.x*model.p*model.z + model.x == 1.0)
model.con3 = Constraint(expr= model.x*model.q*model.z + model.x == 1.0)
# Pyomo differs from AMPL in these cases that involve immutable params (q).
# These never actually become constraints in Pyomo, and for good reason.
model.con4 = Constraint(expr= model.x*model.y*model.z == 1.0)
model.con5 = Constraint(expr= model.x*model.p*model.z == 1.0)
model.con6 = Constraint(rule= simple_constraint_rule(model.x*model.q*model.z == 0.0) )
