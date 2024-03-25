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

from pyomo.environ import ConcreteModel, Var, Param, Objective, Constraint, Binary, Integers, inequality

model = ConcreteModel()
model.x = Var(within=Binary)
model.y = Var(within=Integers)
model.q = Param(initialize=2.0,mutable=True)

model.OBJ = Objective(expr=model.x*model.y)

model.CON1 = Constraint(expr=model.x+model.y <= 2.0*model.q)
model.CON2 = Constraint(expr=model.x+model.y >= model.q)
model.CON3 = Constraint(expr=inequality( model.q, model.x+model.y, 2.0*model.q ))
