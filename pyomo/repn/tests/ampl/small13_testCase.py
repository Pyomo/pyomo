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
# Purpose: For testing to ensure that the Pyomo NL writer properly
#          converts the nonlinear expression to the NL file format.
#
#          This test model relies on the gjh_asl_json executable. It
#          will not solve if sent to a real optimizer.
#

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, maximize

model = ConcreteModel()

model.x = Var(initialize=0.5)

model.obj = Objective(expr=model.x, sense=maximize)

model.c1 = Constraint(expr= (model.x**3 - model.x) == 0)
model.c2 = Constraint(expr= 10*(model.x**3 - model.x) == 0)
model.c3 = Constraint(expr= (model.x**3 - model.x)/10.0 == 0)
