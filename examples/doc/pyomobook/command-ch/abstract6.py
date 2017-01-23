# abstract6.py
from pyomo.environ import *

Model = AbstractModel()

Model.N = Set()
Model.M = Set()
Model.c = Param(Model.N)
Model.a = Param(Model.N, Model.M)
Model.b = Param(Model.M)

Model.x = Var(Model.N, within=NonNegativeReals)

def obj_rule(Model):
    return sum(Model.c[i]*Model.x[i] for i in Model.N)
Model.obj = Objective(rule=obj_rule)

def con_rule(Model, m):
    return sum(Model.a[i,m]*Model.x[i] for i in Model.N) \
                    >= Model.b[m]
Model.con = Constraint(Model.M, rule=con_rule)
