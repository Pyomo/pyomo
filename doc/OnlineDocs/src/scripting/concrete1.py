#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *

model = ConcreteModel()

model.x = Var([1, 2], domain=NonNegativeReals)

model.OBJ = Objective(expr=2 * model.x[1] + 3 * model.x[2])

model.Constraint1 = Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
