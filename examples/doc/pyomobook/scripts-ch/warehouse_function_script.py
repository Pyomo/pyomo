from warehouse_data import *
import pyomo.environ as pe
import warehouse_function as wf

# call function to create model
model = wf.create_wl_model(N, M, d, P)

# solve the model
solver = pe.SolverFactory('glpk')
solver.solve(model)

# look at the solution
model.y.pprint()
