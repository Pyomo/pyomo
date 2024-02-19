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

# An example of how to use the Piecewise component with an
# index set when starting from a discrete set of (x,y)
# points.

from pyomo.core import *

# Note: One or both of these can also be a dictionary of
#       lists with keys matching the index set
x = [0.0, 1.5, 3.0, 5.0]
y = [1.1, -1.1, 2.0, 1.1]

model = ConcreteModel()
model.index = Set(initialize=[1, 2, 3])
model.x = Var(model.index, bounds=(min(x), max(x)))
model.y = Var(model.index)

model.fx = Piecewise(
    model.index, model.y, model.x, pw_pts=x, pw_constr_type='EQ', f_rule=y
)

model.c = ConstraintList()
model.c.add(model.x[1] >= 1.0)
model.c.add(model.x[2] >= 2.0)
model.c.add(model.x[3] >= 3.1)

model.o = Objective(expr=sum_product(model.y))
