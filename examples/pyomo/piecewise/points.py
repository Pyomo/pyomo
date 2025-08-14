#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# An example of how to use the Piecewise component when
# starting from a discrete set of (x,y) points.

import pyomo.environ as pyo

x = [0.0, 1.5, 3.0, 5.0]
y = [1.1, -1.1, 2.0, 1.1]

model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(min(x), max(x)))
model.y = pyo.Var()

model.fx = pyo.Piecewise(model.y, model.x, pw_pts=x, pw_constr_type='EQ', f_rule=y)

model.o = pyo.Objective(expr=model.y)
