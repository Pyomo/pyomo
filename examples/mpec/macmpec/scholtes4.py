# scholtes4.py	LQR2-MN-3-2
# Original Coopr coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An LPEC from S. Scholtes, Judge Inst., University of Cambridge.

# Number of variables:   3 slack
# Number of constraints: 2

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

z_init = {1:0, 2:1}
model.z = Var([1,2], within=NonNegativeReals, initialize=z_init)
model.z3 = Var(initialize=0)

objf = Objective(expr=model.z[1] + model.z[2] - model.z3)

model.lin1 = Constraint(expr=-4 * model.z[1] + model.z3 <= 0)

model.lin2 = Constraint(expr=-4 * model.z[2] + model.z3 <= 0)

model.compl = Complementarity(expr=complements(0 <= model.z[1], model.z[2] >= 0))

