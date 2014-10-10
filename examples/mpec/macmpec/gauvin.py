#
# MPEC example taken from Gauvin and Savard, "Working Paper G9037",
# GERAD, Ecole Polytechnique de Montreal (1992) (first version).
#
#  Let Y(x) := { y | 0 <= y <= 20 - x }.
#  min  f(x,y) := x**2 + (y-10)**2
#  s.t.      0 <= x <= 15
#            y solves MCP((F(y) := 4(x + 2y - 30)), Y(x))
#
# We have modified the problem by adding a dual variable and
# incorporating the constraint y <= 20 - x into the MCP function F.
#
# From a GAMS model by S.P. Dirkse & M.C. Ferris (MPECLIB),
# (see http://www.gams.com/mpec/).
#
# Coopr coding William Hart
# Adapted from AMPL coding Sven Leyffer, University of Dundee, Jan. 2000
#

import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *


model = ConcreteModel()

model.x = Var(bounds=(0,15), initialize=7.5)            # design variable
model.y = Var(within=NonNegativeReals)                  # state variable
model.u = Var(within=NonNegativeReals, initialize=1)    # duals in MCP to determine state vars

model.theta = Objective(expr=model.x**2 + (model.y-10)**2)

# ... F(x,y) from original problem
model.Fy = Complementarity(expr=complements(0 <= 4 * (model.x + 2*model.y - 30) + model.u, model.y >= 0))

model.Fu = Complementarity(expr=complements(0 <=  20 - model.x - model.y, model.u >= 0))

