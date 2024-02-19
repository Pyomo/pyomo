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

# An example of how to use the Piecewise component when
# starting from a discrete set of (x,y) points.

from pyomo.core import *

x = [0.0, 1.5, 3.0, 5.0]
y = [1.1, -1.1, 2.0, 1.1]

model = ConcreteModel()
model.x = Var(bounds=(min(x), max(x)))
model.y = Var()

model.fx = Piecewise(model.y, model.x, pw_pts=x, pw_constr_type='EQ', f_rule=y)

model.o = Objective(expr=model.y)
