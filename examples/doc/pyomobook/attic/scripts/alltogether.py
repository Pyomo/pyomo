import pyomo.environ
from pyomo.opt import SolverFactory
from pyutilib.misc import Options

import sys
import math
from pyomo.core import*

LenIn = 5
WidthIn = 4
pIn = 1
qIn = 1

model = ConcreteModel()


# Parameters
model.length = Param(within=Reals, initialize=LenIn)
model.width = Param(within=Reals, initialize=WidthIn)
model.p = Param(within=Reals, initialize=pIn)
model.q = Param(within=Reals, initialize=qIn)

# Variables
model.x = Var(bounds=(-LenIn, LenIn), initialize=0)
model.y = Var(bounds=(-WidthIn, WidthIn), initialize=0)


# Objective
#model.obj = Objective(expr = math.sqrt(((model.p - model.x)**2) + ((model.q - model.y)**2)))
model.obj = Objective(expr = (((model.p - model.x)**2) + ((model.q - model.y)**2))**0.5)

# Constraints
model.KeineAhnung = Constraint(expr = ((model.x / model.length)**2) + ((model.y / model.width)**2) - 1 >= 0)

model.pprint()

model.skip_canonical_repn = True # for nonlinear models

instance=model.create()

SolverName = "asl"
so = Options()
so.solver = "ipopt"
opt=SolverFactory(SolverName, options=so)

if opt is None:
    print("Could not construct solver %s : %s" % (SolverName, so.solver))
    sys.exit(1)

results=opt.solve(instance)
results.write()
instance.load(results) # put results in model

# because we know there is a variable named x
x_var = getattr(instance, "x")
x_val = x_var()

print("x was "+str(x_val))
