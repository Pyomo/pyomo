from pyomo.environ import *         # import pyomo environment
from wl_concrete import model, N, M # import model and sets

solver = SolverFactory('glpk') # create the glpk solver
solver.solve(model)            # solve 

# @output:
# produce nicely formatted output
for wl in N:
    if value(model.y[wl]) > 0.5:
        customers = [str(cl) for cl in M if value(model.x[wl, cl] > 0.5)]
        print(str(wl)+' serves customers: '+str(customers))
    else:
        print(str(wl)+": do not build")
# @:output
