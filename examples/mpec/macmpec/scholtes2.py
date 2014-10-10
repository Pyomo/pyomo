# scholtes2.py	QOR2-MN-4-2
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An MPEC from S. Scholtes, Research Papers in Management Studies, 26/1997,
# The Judge Institute, University of Cambridge, England.

# Number of variables:   3 
# Number of constraints: 1

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var(within=NonNegativeReals, initialize=1)
model.y = Var([1,2], initialize=1)

model.f = Objective(expr=(model.x + 1)**2 + model.y[1]**2 + 10*(model.y[2] + 1)**2)

model.lin_cs = Constraint(expr=model.y[2] >= 0)

model.nln_cs = Complementarity(expr=complements(0 <= -exp(model.x) + model.y[1] - exp(model.y[2]), model.x >= 0))

