from pyomo.environ import *
from simple import model

# Construct solver object
# @solver:
solver = SolverFactory('asl:ipopt')
# @:solver

# Apply solver and load results into model
results = solver.solve(model)

# Store results into results object
model.solutions.store_to(results)

# Print results
print(results)
