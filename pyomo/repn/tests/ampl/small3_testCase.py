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
#          NL writer properly reports the values corresponding
#          to the nl file header line with the label
#          '# nonlinear vars in constraints, objectives, both'
#

from pyomo.environ import *

model = ConcreteModel()

model.x = Var(initialize=1.0)
model.y = Var(initialize=1.0)

model.OBJ = Objective(expr=model.x*model.y)

model.CON1 = Constraint(expr=model.y**2 == 4)

