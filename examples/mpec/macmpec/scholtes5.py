# scholtes5.py	QBR-AN-LCP-3-2-2 
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# A QPEC from S. Scholtes, Judge Inst., University of Cambridge.
# see S. Scholtes, "Convergence properties of a regularization
# scheme for MPCCs", SIAM J. Optimization 11(4):918-936, 2001.

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.z = Var([1,2,3], within=NonNegativeReals, initialize=1)

model.objf = Objective(expr=(model.z[1] - 1)**2 + (model.z[2] - 2)**2 + (model.z[3] + 1)**2)

model.compl1 = Complementarity(expr=complements(0 <= model.z[1], model.z[3] >= 0))

model.compl2 = Complementarity(expr=complements(0 <= model.z[2], model.z[3] >= 0))

