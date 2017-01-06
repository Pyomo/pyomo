# @script:
from warehouse_data import *
import pyomo.environ as pe
import warehouse_function as wf

# call function to create model
model = wf.create_wl_model(N, M, d, P)

# solve the model
solver = pe.SolverFactory('glpk')
solver_opt = dict()
solver_opt['log'] = 'warehouse.log'
solver_opt['nointopt'] = None
solver.solve(model, options=solver_opt)

# look at the solution
model.y.pprint()
# @:script

import os
os.remove('warehouse.log')
