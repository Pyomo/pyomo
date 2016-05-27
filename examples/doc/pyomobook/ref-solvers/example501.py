from pyomo.environ import *
from simple import model

# @run:
m1 = ConcreteModel()
m1.x = Var(within=NonNegativeReals)
m1.y = Var(within=NonNegativeReals)
m1.o = Objective(expr=m1.x+3*m1.y)
m1.c = Constraint(expr=m1.x+m1.y == 1)

solver = SolverFactory("glpk")
results = solver.solve(m1)
m1.solutions.store_to(results)

m2 = ConcreteModel()
m2.x = Var(within=NonNegativeReals)

# Load just the x variable
m2.solutions.load_from(results, ignore_invalid_labels=True)

# @:run

m2.display()
