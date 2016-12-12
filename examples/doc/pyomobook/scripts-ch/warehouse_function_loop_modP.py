from warehouse_data import *
from pyomo.environ import *

# create the model
model = ConcreteModel(name="(WL)")
model.P = Param(initialize=P, mutable=True)
model.x = Var(N, M, bounds=(0,1))
model.y = Var(N, within=Binary)

def obj_rule(model):
    return sum(d[n,m]*model.x[n,m] for n in N for m in M)
model.obj = Objective(rule=obj_rule)

def one_per_cust_rule(model, m):
    return sum(model.x[n,m] for n in N) == 1
model.one_per_cust = Constraint(M, rule=one_per_cust_rule)

def warehouse_active_rule(model, n, m):
    return model.x[n,m] <= model.y[n]
model.warehouse_active = Constraint(N, M, rule=warehouse_active_rule)

def num_warehouses_rule(model):
    return sum(model.y[n] for n in N) <= model.P
model.num_warehouses = Constraint(rule=num_warehouses_rule)

# execute the loop
for pp in [1,2,3]:
    # change the value of the parameter P
    model.P = pp
    
    # solve the model
    solver = SolverFactory('glpk')
    solver.solve(model)

    # look at the solution
    print('--- P = {0} ---'.format(pp))
    model.y.pprint()
