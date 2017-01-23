from pyomo.environ import *

def create_wl_model(N, M, d, P):
    # create the model
    model = ConcreteModel(name="(WL)")
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
        return sum(model.y[n] for n in N) <= P
    model.num_warehouses = Constraint(rule=num_warehouses_rule)

    return model
