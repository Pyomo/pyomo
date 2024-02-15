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

# scholtes4.py	LQR2-MN-3-2
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer

# An LPEC from S. Scholtes, Judge Inst., University of Cambridge.

# Number of variables:   3 slack
# Number of constraints: 2

from pyomo.environ import *
from pyomo.mpec import *


model = ConcreteModel()

z_init = {1: 0, 2: 1}
model.z = Var([1, 2], within=NonNegativeReals, initialize=z_init)
model.z3 = Var(initialize=0)

model.objf = Objective(expr=model.z[1] + model.z[2] - model.z3)

model.lin1 = Constraint(expr=-4 * model.z[1] + model.z3 <= 0)

model.lin2 = Constraint(expr=-4 * model.z[2] + model.z3 <= 0)

model.compl = Complementarity(expr=complements(0 <= model.z[1], model.z[2] >= 0))
