#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.core import ConcreteModel, Var, Objective, maximize

model = ConcreteModel()

model.x = Var()
model.y = Var()

model.o = Objective(expr=-model.x-model.y, sense=maximize)
