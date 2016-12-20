from pyomo.environ import *

model = ConcreteModel()
model.x = Var(bounds=(0,5))
model.y = Var(bounds=(0,1))
model.con = Constraint(expr=model.x + model.y == 1.0)
model.obj = Objective(expr=model.y-model.x)

# solve the problem
# @solver:
solver = SolverFactory('glpk')
# @:solver
solver.solve(model)
print(value(model.x)) # 1.0
print(value(model.y)) # 0.0

# add a constraint
model.con2 = Constraint(expr=4.0*model.x + model.y == 2.0)
solver.solve(model)
print(value(model.x)) # 0.33
print(value(model.y)) # 0.66

# deactivate a constraint
model.con.deactivate()
solver.solve(model)
print(value(model.x)) # 0.5
print(value(model.y)) # 0.0

# activate a constraint
model.con.activate()
solver.solve(model)
print(value(model.x)) # 0.33
print(value(model.y)) # 0.66

# delete a constraint
del model.con2
solver.solve(model)
print(value(model.x)) # 1.0
print(value(model.y)) # 0.0

# fix the variable
model.x.fix(0.5)
solver.solve(model)
print(value(model.x)) # 0.5
print(value(model.y)) # 0.5

# unfix the variable
model.x.unfix()
solver.solve(model)
print(value(model.x)) # 1.0
print(value(model.y)) # 0.0

