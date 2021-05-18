import json
import pyomo.environ as pyo
from warehouse_model import create_wl_model

# load the data from a json file
with open('warehouse_data.json', 'r') as fd:
    data = json.load(fd)

# call function to create model
model = create_wl_model(data, P=2)

# solve the model
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# print the expression for the objective function
print(model.obj.expr)

# print the value of the objective function
# at the solution
print(pyo.value(model.obj))

# print the value of a particular variable
print(pyo.value(model.y['Harlingen']))




