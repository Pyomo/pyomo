import sys
import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.common.collections import Options

from nonlin import model

model.pprint()

model.skip_canonical_repn = True # for nonlinear models

instance=model.create()

SolverName = "asl"
so = Options()
so.solver = "ipopt"
opt=SolverFactory(SolverName, options=so)

if opt is None:
    print("Could not construct solver %s:%s" % (SolverName,so.solver))
    sys.exit(1)

results=opt.solve(instance)
results.write()
instance.load(results) # put results in model

# because we know there is a variable named x
x_var = getattr(instance, "x")
x_val = x_var()

print("x was "+str(x_val))
