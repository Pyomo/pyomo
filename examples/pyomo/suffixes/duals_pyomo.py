# A Suffix example for the collection of duals.

from pyomo.core import *

### Create a trivial and infeasible example model
model = ConcreteModel()
model.x = Var(within=NonNegativeReals)
model.obj = Objective(expr=model.x)
model.con = Constraint(expr=model.x >= 1)
###

# Declare an IMPORT Suffix to store the dual information that will
# be returned by the solver. When Suffix components are declared 
# with an IMPORT direction, Pyomo solver interfaces will attempt to collect
# this named information from a solver solution.
model.dual = Suffix(direction=Suffix.IMPORT)

