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
#          NL writer properly modifies product expressions
#          with only constant terms in the denominator (that
#          are involved in nonlinear expressions).
#          The ASL differentiation routines seem to have a
#          bug that causes the lagrangian hessian to become
#          dense unless this constant term in moved to the
#          numerator.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import ConcreteModel, Var, Param, Objective, Constraint

model = ConcreteModel()
model.x = Var(bounds=(-1.0,1.0),initialize=1.0)
model.y = Var(bounds=(-1.0,1.0),initialize=2.0)
model.v = Var(bounds=(-1.0,1.0),initialize=3.0)
model.p = Param(initialize=2.0)
model.q = Param(initialize=2.0,mutable=True)

model.OBJ = Objective(expr=model.x**2/model.p + model.x**2/model.q)
model.CON1 = Constraint(expr=1.0/model.p*model.v*(model.x-model.y) == 2.0)
model.CON2 = Constraint(expr=model.v*1.0/model.p*(model.x-model.y) == 2.0)
model.CON3 = Constraint(expr=model.v*(model.x-model.y)/model.p == 2.0)
model.CON4 = Constraint(expr=model.v*(model.x/model.p-model.y/model.p) == 2.0)
model.CON5 = Constraint(expr=model.v*(model.x-model.y)*(1.0/model.p) == 2.0)
model.CON6 = Constraint(expr=model.v*(model.x-model.y) == 2.0*model.p)

model.CON7 = Constraint(expr=1.0/model.q*model.v*(model.x-model.y) == 2.0)
model.CON8 = Constraint(expr=model.v*1.0/model.q*(model.x-model.y) == 2.0)
model.CON9 = Constraint(expr=model.v*(model.x-model.y)/model.q == 2.0)
model.CON10 = Constraint(expr=model.v*(model.x/model.q-model.y/model.q) == 2.0)
model.CON11 = Constraint(expr=model.v*(model.x-model.y)*(1.0/model.q) == 2.0)
model.CON12 = Constraint(expr=model.v*(model.x-model.y) == 2.0*model.q)
