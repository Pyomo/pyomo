from pyomo.environ import *   # import pyomo environment
from wl_concrete import model # import model

solver = SolverFactory('glpk') # create the glpk solver
solver.solve(model)            # solve 

# @output:
model.y.pprint() # print the optimal warehouse locations
# @:output
