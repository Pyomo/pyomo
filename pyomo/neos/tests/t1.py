#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import ConcreteModel, Var, Objective, Constraint, maximize

M = ConcreteModel()
M.x = Var(bounds=(0,1))
M.o = Objective(expr=2*M.x, sense=maximize)
M.c = Constraint(expr=M.x <= 0.5)

model = M

