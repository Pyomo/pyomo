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

# wl_list.py: Warehouse location problem using constraint lists
import pyomo.environ as pyo

model = pyo.ConcreteModel(name="(PM)")

# @data:
N = ['Harlingen', 'Memphis', 'Ashland']
M = ['NYC', 'LA', 'Chicago', 'Houston']
# @:data
d = {
    ('Harlingen', 'NYC'): 1956,
    ('Harlingen', 'LA'): 1606,
    ('Harlingen', 'Chicago'): 1410,
    ('Harlingen', 'Houston'): 330,
    ('Memphis', 'NYC'): 1096,
    ('Memphis', 'LA'): 1792,
    ('Memphis', 'Chicago'): 531,
    ('Memphis', 'Houston'): 567,
    ('Ashland', 'NYC'): 485,
    ('Ashland', 'LA'): 2322,
    ('Ashland', 'Chicago'): 324,
    ('Ashland', 'Houston'): 1236,
}
P = 2
# @vars:
model.x = pyo.Var(N, M, bounds=(0, 1))
model.y = pyo.Var(N, within=pyo.Binary)
# @:vars

# @obj:
model.obj = pyo.Objective(expr=sum(d[n, m] * model.x[n, m] for n in N for m in M))
# @:obj

# @conslist:
model.demand = pyo.ConstraintList()
for m in M:
    model.demand.add(sum(model.x[n, m] for n in N) == 1)

model.warehouse_active = pyo.ConstraintList()
for n in N:
    for m in M:
        model.warehouse_active.add(model.x[n, m] <= model.y[n])
# @:conslist

# @scalarcon:
model.num_warehouses = pyo.Constraint(expr=sum(model.y[n] for n in N) <= P)
# @:scalarcon

model.pprint()
