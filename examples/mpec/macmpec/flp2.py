# flp2.py  QLR-AN-LCP-4-2-2
# 
# Problem 2 from Fukushima, M. Luo, Z.-Q.Pang, J.-S.,
# "A globally convergent Sequential Quadratic Programming
# Algorithm for Mathematical Programs with Linear Complementarity 
# Constraints", Computational Optimization and Applications, 10(1),
# pp. 5-34, 1998.
#
# Note: Problem 1 is equivalent to stackelberg1.mod 

import pyomo.modeling
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.x = Var([1,2], bounds=(0,10))
model.y = Var([1,2], within=NonNegativeReals)

model.objf = Objective(expr=0.5*( (model.x[1]+model.x[2]+model.y[1]-15)**2 + (model.x[1]+model.x[2]+model.y[2]-15)**2 ))

model.compl1 = Complementarity(expr=complements(0 <= model.y[1],
                    0 <= 8/3*model.x[1] + 2*model.x[2] + 2*model.y[1] + 8/3*model.y[2] - 36))

model.compl2 = Complementarity(expr=complements(0 <= model.y[2],
                    0 <= 2*model.x[1] + 5/4*model.x[2] + 5/4*model.y[1] + 2*model.y[2] - 25))

