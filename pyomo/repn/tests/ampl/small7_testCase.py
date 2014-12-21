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
#          are involved in nonlinear expressions).
#          The ASL differentiation routines seem to have a 
#          bug that causes the lagrangian hessian to become
#          dense unless this constant term in moved to the 
#          numerator. 
#
#          This test model relies on the asl_test executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import *

model = ConcreteModel()
model.x = Var(bounds=(-1.0,1.0),initialize=1.0)
model.y = Var(bounds=(-1.0,1.0),initialize=2.0)
model.v = Var(bounds=(-1.0,1.0),initialize=3.0)
model.p = Var(initialize=2.0)
model.p.fixed = True
model.q = Param(initialize=2.0)

model.OBJ = Objective(expr=model.x)
model.CON1a = Constraint(expr=1.0/model.p/model.q*model.v*(model.x-model.y) == 2.0)
model.CON2a = Constraint(expr=model.v*1.0/model.p/model.p*(model.x-model.y) == 2.0)
model.CON3a = Constraint(expr=model.v*(model.x-model.y)/model.p/model.q == 2.0)
model.CON4a = Constraint(expr=model.v*(model.x/model.p/model.q-model.y/model.p/model.q) == 2.0)
model.CON5a = Constraint(expr=model.v*(model.x-model.y)*(1.0/model.p/model.q) == 2.0)
model.CON6a = Constraint(expr=model.v*(model.x-model.y) == 2.0*model.p*model.q)

model.CON1b = Constraint(expr=1.0/(model.p*model.q)*model.v*(model.x-model.y) == 2.0)
model.CON2b = Constraint(expr=model.v*1.0/(model.p*model.p)*(model.x-model.y) == 2.0)
model.CON3b = Constraint(expr=model.v*(model.x-model.y)/(model.p*model.q) == 2.0)
model.CON4b = Constraint(expr=model.v*(model.x/(model.p*model.q)-model.y/(model.p*model.q)) == 2.0)
model.CON5b = Constraint(expr=model.v*(model.x-model.y)*(1.0/(model.p*model.q)) == 2.0)
model.CON6b = Constraint(expr=model.v*(model.x-model.y) == 2.0*(model.p*model.q))

model.CON1c = Constraint(expr=1.0/(model.p+model.q)*model.v*(model.x-model.y) == 2.0)
model.CON2c = Constraint(expr=model.v*1.0/(model.p+model.p)*(model.x-model.y) == 2.0)
model.CON3c = Constraint(expr=model.v*(model.x-model.y)/(model.p+model.q) == 2.0)
model.CON4c = Constraint(expr=model.v*(model.x/(model.p+model.q)-model.y/(model.p+model.q)) == 2.0)
model.CON5c = Constraint(expr=model.v*(model.x-model.y)*(1.0/(model.p+model.q)) == 2.0)
model.CON6c = Constraint(expr=model.v*(model.x-model.y) == 2.0*(model.p+model.q))

model.CON1d = Constraint(expr=1.0/((model.p+model.q)**2)*model.v*(model.x-model.y) == 2.0)
model.CON2d = Constraint(expr=model.v*1.0/((model.p+model.p)**2)*(model.x-model.y) == 2.0)
model.CON3d = Constraint(expr=model.v*(model.x-model.y)/((model.p+model.q)**2) == 2.0)
model.CON4d = Constraint(expr=model.v*(model.x/((model.p+model.q)**2)-model.y/((model.p+model.q)**2)) == 2.0)
model.CON5d = Constraint(expr=model.v*(model.x-model.y)*(1.0/((model.p+model.q)**2)) == 2.0)
model.CON6d = Constraint(expr=model.v*(model.x-model.y) == 2.0*((model.p+model.q)**2))

