# buildactions.py: Warehouse location problem showing build actions
from pyomo.environ import *

model = AbstractModel()

model.N = Set() # Set of warehouses
model.M = Set() # Set of customers
model.d = Param(model.N,model.M)
model.P = Param()

model.x = Var(model.N, model.M, bounds=(0,1))
model.y = Var(model.N, within=Binary)

def checkPN_rule(model):
    return model.P <= len(model.N)
model.checkPN = BuildCheck(rule=checkPN_rule)

def obj_rule(model):
    return sum(model.d[n,m]*model.x[n,m] for n in model.N for m in model.M)
model.obj = Objective(rule=obj_rule)

def one_per_cust_rule(model, m):
    return sum(model.x[n,m] for n in model.N) == 1
model.one_per_cust = Constraint(model.M, rule=one_per_cust_rule)

def warehouse_active_rule(model, n, m):
    return model.x[n,m] <= model.y[n]
model.warehouse_active = Constraint(model.N, model.M, rule=warehouse_active_rule)

def num_warehouses_rule(model):
    return sum(model.y[n] for n in model.N) <= model.P
model.num_warehouses = Constraint(rule=num_warehouses_rule)

def printM_rule(model):
    model.M.pprint()
model.printM = BuildAction(rule=printM_rule)
