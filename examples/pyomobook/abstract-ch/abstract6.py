# abstract6.py
import pyomo.environ as pyo

Model = pyo.AbstractModel()

Model.N = pyo.Set()
Model.M = pyo.Set()
Model.c = pyo.Param(Model.N)
Model.a = pyo.Param(Model.N, Model.M)
Model.b = pyo.Param(Model.M)

Model.x = pyo.Var(Model.N, within=pyo.NonNegativeReals)

def obj_rule(Model):
    return sum(Model.c[i]*Model.x[i] for i in Model.N)
Model.obj = pyo.Objective(rule=obj_rule)

def con_rule(Model, m):
    return sum(Model.a[i,m]*Model.x[i] for i in Model.N) \
                    >= Model.b[m]
Model.con = pyo.Constraint(Model.M, rule=con_rule)
