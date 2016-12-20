from pyomo.environ import *

model = ConcreteModel()
model.x = Var()
model.o = Objective(expr= model.x)
model.c = Constraint(expr= model.x >= 1)

solver = SolverFactory("glpk")
results = solver.solve(model)

print(results)
