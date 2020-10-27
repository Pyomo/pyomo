#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Var, Objective, Constraint, Integers

model = ConcreteModel()

model.x = Var(bounds=(1,None), within=Integers)
model.y = Var(bounds=(1,None), within=Integers)

model.o = Objective(expr=model.x-model.x)

model.c = Constraint(expr=model.x+model.y >= 3)
