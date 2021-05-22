import json
import pyomo.environ as pyo
from warehouse_model import create_wl_model

# load the data from a json file
with open('warehouse_data.json', 'r') as fd:
    data = json.load(fd)

# call function to create model
model = create_wl_model(data, P=2)

# create the solver
solver = pyo.SolverFactory('glpk')

print("Initial values for X")
model.x.pprint()

# @load_solutions:
from pyomo.opt import SolverStatus, TerminationCondition
# Wait to load the solution into the model until
# after the solver status is checked
results = solver.solve(model, load_solutions=False)
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    # Manually load the solution into the model
    model.solutions.load_from(results)
else:
    print("Solve failed.")
# @:load_solutions

print("Solution for X")
model.x.pprint()
