# wl_concrete.py: ConcreteModel version of warehouse location determination problem
from pyomo.environ import *

model = ConcreteModel(name="(WL)")

N = ['Harlingen', 'Memphis', 'Ashland']
M = ['NYC', 'LA', 'Chicago', 'Houston']
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

model.x = Var(N, M, bounds=(0,1))
model.y = Var(N, within=Binary)

def obj_rule(model):
    return sum(d[n,m]*model.x[n,m] for n in N for m in M)
model.obj = Objective(rule=obj_rule)

# @deliver:
def one_per_cust_rule(model, m):
    return sum(model.x[n,m] for n in N) == 1
model.one_per_cust = Constraint(M, rule=one_per_cust_rule)
# @:deliver

def warehouse_active_rule(model, n, m):
    return model.x[n,m] <= model.y[n]
model.warehouse_active = Constraint(N, M, rule=warehouse_active_rule)

def num_warehouses_rule(model):
    return sum(model.y[n] for n in N) <= P
model.num_warehouses = Constraint(rule=num_warehouses_rule)
