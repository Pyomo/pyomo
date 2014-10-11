# ralph1.py	LUR-AN-LCP-2-0-1 
# Original Pyomo coding by William Hart
# Adapted from original AMPL coding by Sven Leyffer

# An LPEC from D. Ralph, Judge Inst., University of Cambridge.
# This problem violates strong stationarity, but is B-stationary.

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)

model.f1 = Objective(expr=2*model.x - model.y)
#minimize f2:   x - y;

model.compl = Complementarity(expr=complements(0 <= model.y, model.y-model.x >= 0))

