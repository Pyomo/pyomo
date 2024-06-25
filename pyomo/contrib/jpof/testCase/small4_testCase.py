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
#          NL writer properly reports the values corresponding
#          to the nl file header line with the label
#          '# nonlinear vars in constraints, objectives, both'
#


from pyomo.environ import ConcreteModel, Var, Objective, Constraint

model = ConcreteModel()

model.x = Var(initialize=1.0)
model.y = Var(initialize=1.0)

model.OBJ = Objective(expr=model.y**2)

model.CON1 = Constraint(expr=model.y*model.x == 4)

