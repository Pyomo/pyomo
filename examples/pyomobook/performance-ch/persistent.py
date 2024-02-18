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

# @model:
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.x = pyo.Var()
m.y = pyo.Var()
m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
m.c = pyo.Constraint(expr=m.y >= -2 * m.x + 5)
# @:model

# @creation:
opt = pyo.SolverFactory('gurobi_persistent')
# @:creation

# @set_instance:
opt.set_instance(m)
# @:set_instance

# @solve:
results = opt.solve()
# @:solve

print('Objective after solve 1: ', pyo.value(m.obj))

# @add_constraint:
m.c2 = pyo.Constraint(expr=m.y >= m.x)
opt.add_constraint(m.c2)
# @:add_constraint

# @solve2:
results = opt.solve()
# @:solve2

print('Objective after solve 2: ', pyo.value(m.obj))

# @remove_constraint:
opt.remove_constraint(m.c2)
del m.c2
results = opt.solve()
# @:remove_constraint

print('Objective after solve 3: ', pyo.value(m.obj))

# @extra_constraint:
m = pyo.ConcreteModel()
m.x = pyo.Var()
m.y = pyo.Var()
m.c = pyo.Constraint(expr=m.y >= -2 * m.x + 5)
opt = pyo.SolverFactory('gurobi_persistent')
opt.set_instance(m)
# WRONG:
del m.c
m.c = pyo.Constraint(expr=m.y <= m.x)
opt.add_constraint(m.c)
# @:extra_constraint

# @no_extra_constraint:
m = pyo.ConcreteModel()
m.x = pyo.Var()
m.y = pyo.Var()
m.c = pyo.Constraint(expr=m.y >= -2 * m.x + 5)
opt = pyo.SolverFactory('gurobi_persistent')
opt.set_instance(m)
# Correct:
opt.remove_constraint(m.c)
del m.c
m.c = pyo.Constraint(expr=m.y <= m.x)
opt.add_constraint(m.c)
# @:no_extra_constraint

# @update_var:
m = pyo.ConcreteModel()
m.x = pyo.Var()
m.y = pyo.Var()
m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
m.c = pyo.Constraint(expr=m.y >= -2 * m.x + 5)
opt = pyo.SolverFactory('gurobi_persistent')
opt.set_instance(m)
m.x.setlb(1.0)
opt.update_var(m.x)
# @:update_var

# @indexed_component:
m.v = pyo.Var([0, 1, 2])
m.c2 = pyo.Constraint([0, 1, 2])
for i in range(3):
    m.c2[i] = m.v[i] == i
for v in m.v.values():
    opt.add_var(v)
for c in m.c2.values():
    opt.add_constraint(c)
# @:indexed_component

# @save_results:
m = pyo.ConcreteModel()
m.x = pyo.Var()
m.y = pyo.Var()
m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
m.c = pyo.Constraint(expr=m.y >= -2 * m.x + 5)
opt = pyo.SolverFactory('gurobi_persistent')
opt.set_instance(m)
results = opt.solve(save_results=False)
# @:save_results

print('Objective after solve 4: ', pyo.value(m.obj))

# @load_from:
results = opt.solve(save_results=False, load_solutions=False)
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    try:
        m.solutions.load_from(results)
    except AttributeError:
        print('AttributeError was raised')
# @:load_from

# @load_vars:
results = opt.solve(save_results=False, load_solutions=False)
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    opt.load_vars()
# @:load_vars

# @load_vars2:
results = opt.solve(save_results=False, load_solutions=False)
if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    opt.load_vars([m.x])
# @:load_vars2

print('Objective after solve 5: ', pyo.value(m.obj))
