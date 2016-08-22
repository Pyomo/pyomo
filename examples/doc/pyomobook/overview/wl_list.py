# wl_list.py: Warehouse location problem using constraint lists
from pyomo.environ import *

model = ConcreteModel(name="(PM)")

# @data:
N = ['Harlingen', 'Memphis', 'Ashland']
M = ['NYC', 'LA', 'Chicago', 'Houston']
# @:data
d = {('Harlingen', 'NYC'): 1956, \
     ('Harlingen', 'LA'): 1606, \
     ('Harlingen', 'Chicago'): 1410, \
     ('Harlingen', 'Houston'): 330, \
     ('Memphis', 'NYC'): 1096, \
     ('Memphis', 'LA'): 1792, \
     ('Memphis', 'Chicago'): 531, \
     ('Memphis', 'Houston'): 567, \
     ('Ashland', 'NYC'): 485, \
     ('Ashland', 'LA'): 2322, \
     ('Ashland', 'Chicago'): 324, \
     ('Ashland', 'Houston'): 1236 }
P = 2
# @vars:
model.x = Var(N, M, bounds=(0,1))
model.y = Var(N, within=Binary)
# @:vars

# @obj:
model.obj = Objective(expr=sum(d[n,m]*model.x[n,m] for n in N for m in M))
# @:obj

# @conslist:
model.one_per_cust = ConstraintList()
for m in M:
    model.one_per_cust.add(sum(model.x[n,m] for n in N) == 1)

model.warehouse_active = ConstraintList()
for n in N:
    for m in M:
        model.warehouse_active.add(model.x[n,m] <= model.y[n])
# @:conslist

# @scalarcon:
model.num_warehouses = Constraint(expr=sum(model.y[n] for n in N) <= P)
# @:scalarcon

model.pprint()
