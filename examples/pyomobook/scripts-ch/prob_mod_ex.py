#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.ConcreteModel()
model.x = pyo.Var(bounds=(0, 5))
model.y = pyo.Var(bounds=(0, 1))
model.con = pyo.Constraint(expr=model.x + model.y == 1.0)
model.obj = pyo.Objective(expr=model.y - model.x)

# solve the problem
# @solver:
solver = pyo.SolverFactory('glpk')
# @:solver
solver.solve(model)
print(pyo.value(model.x))  # 1.0
print(pyo.value(model.y))  # 0.0

# add a constraint
model.con2 = pyo.Constraint(expr=4.0 * model.x + model.y == 2.0)
solver.solve(model)
print(pyo.value(model.x))  # 0.33
print(pyo.value(model.y))  # 0.66

# deactivate a constraint
model.con.deactivate()
solver.solve(model)
print(pyo.value(model.x))  # 0.5
print(pyo.value(model.y))  # 0.0

# re-activate a constraint
model.con.activate()
solver.solve(model)
print(pyo.value(model.x))  # 0.33
print(pyo.value(model.y))  # 0.66

# delete a constraint
del model.con2
solver.solve(model)
print(pyo.value(model.x))  # 1.0
print(pyo.value(model.y))  # 0.0

# fix a variable
model.x.fix(0.5)
solver.solve(model)
print(pyo.value(model.x))  # 0.5
print(pyo.value(model.y))  # 0.5

# unfix a variable
model.x.unfix()
solver.solve(model)
print(pyo.value(model.x))  # 1.0
print(pyo.value(model.y))  # 0.0
