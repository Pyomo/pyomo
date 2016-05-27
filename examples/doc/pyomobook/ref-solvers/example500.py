from pyomo.environ import *
from simple import model

# Construct solver object
solver = SolverFactory('glpk')

# @load1:
# Apply solver and load the second solution
results = solver.solve(model, select=1)
# @:load1

print(results)

# @load2:
# Apply solver and do not load results
results = solver.solve(model, load_solutions=False)
# @:load2

print(results)

# @load3:
# Apply the solver
results = solver.solve(model, load_solutions=False)
# Load the second solution
model.solutions.load_from(results, select=2)
# @:load3

print(results)

# @load4:
# Apply the solver and load the first solution
results = solver.solve(model)
# Load the second solution
model.solutions.select(1)
# @:load4

print(results)

# @load5:
# Apply the solver
results = solver.solve(model)
# Store the variable values into the results object
model.store_to(results)
# @:load5

print(results)


