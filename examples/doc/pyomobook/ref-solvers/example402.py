from pyomo.environ import *
from simple import model

# Construct solver object
solver = SolverFactory('gurobi')

# Apply solver and load results into model
results = solver.solve(model, solver_io='nl')

# Store results into results object
model.solutions.store_to(results)

# Print results
print(results)
