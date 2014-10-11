# outrata32.py QUR-AN-NCP-5-0-4
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, University of Dundee

# An MPEC from S. Scholtes, Research Papers in Management Studies, 26/1997,
# The Judge Institute, University of Cambridge, England.
# See also Outrata, SIAM J. Optim. 4(2), pp.340ff, 1994.

# Number of variables:   5 
# Number of constraints: 4

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var(range(1,5), within=NonNegativeReals)
model.y = Var(bounds=(0,10))

model.f = Objective(expr=( (model.x[1] - 3)**2 + (model.x[2] - 4)**2 + (model.x[3] - 1)**2)/2)

model.nlcs1 = Complementarity(expr=complements(0 <= (1 + 0.2*model.y)*model.x[1] - (3 + 1.333*model.y) - 0.333*model.x[3] + 2*model.x[1]*model.x[4], model.x[1] >= 0))

model.nlcs2 = Complementarity(expr=complements(0 <= (1 + 0.2*model.y)*model.x[2] - model.y + model.x[3] + 2*model.x[2]*model.x[4], model.x[2] >= 0))

model.nlcs3 = Complementarity(expr=complements(0 <= 0.333*model.x[1] - model.x[2] + 1 - 0.1*model.y, model.x[3] >= 0))

model.nlcs4 = Complementarity(expr=complements(0 <= 9 + 0.1*model.y - model.x[1]**2 - model.x[2]**2, model.x[4] >= 0))

