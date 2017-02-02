from pyomo.environ import *

model = ConcreteModel()

model.x = Var(within=NonNegativeReals)
model.y = Var(within=NonNegativeReals)
model.obj = Objective(expr=model.x + 2*model.y)
model.con1 = Constraint(expr=3*model.x + 4*model.y >= 1)
model.con2 = Constraint(expr=2*model.x + 5*model.y >= 2)

# @solver_io:
# Construct solver object
solver = SolverFactory('gurobi')

# Apply solver and load results into model
solver.solve(model, solver_io='nl')
# @:solver_io
