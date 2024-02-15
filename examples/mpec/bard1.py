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

# bard1.py QQR2-MN-8-5
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An MPEC from J.F. Bard, Convex two-level optimization,
# Mathematical Programming 40(1), 15-27, 1988.

# Number of variables:   2 + 3 slack + 3 multipliers
# Number of constraints: 4

from pyomo.environ import *
from pyomo.mpec import *


model = ConcreteModel()
model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

# ... multipliers
model.l = Var([1, 2, 3])

model.f = Objective(expr=(model.x - 5) ** 2 + (2 * model.y + 1) ** 2)

model.KKT = Constraint(
    expr=2 * (model.y - 1) - 1.5 * model.x + model.l[1] - model.l[2] * 0.5 + model.l[3]
    == 0
)

model.lin_1 = Complementarity(
    expr=complements(0 <= 3 * model.x - model.y - 3, model.l[1] >= 0)
)
model.lin_2 = Complementarity(
    expr=complements(0 <= -model.x + 0.5 * model.y + 4, model.l[2] >= 0)
)
model.lin_3 = Complementarity(
    expr=complements(0 <= -model.x - model.y + 7, model.l[3] >= 0)
)
