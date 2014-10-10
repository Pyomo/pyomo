#
# Author:  Gabe Hackebeil
# Purpose: For regression testing to ensure that the Pyomo
#          NL writer properly reports the values corresponding
#          to the nl file header line with the label
#          '# nonlinear vars in constraints, objectives, both'
#

from coopr.pyomo import *

model = ConcreteModel()

model.x = Var(initialize=1.0)
model.y = Var(initialize=1.0)

model.OBJ = Objective(expr=model.x**2)

model.CON1 = Constraint(expr=model.y**2 == 4)

