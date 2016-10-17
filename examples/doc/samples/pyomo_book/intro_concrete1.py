from pyomo.environ import *
import mydata

model = ConcreteModel()

model.x = Var(mydata.N, within=NonNegativeReals)

def obj_rule(model):
    return sum(mydata.c[i]*model.x[i] for i in mydata.N)
model.obj = Objective(rule=obj_rule)

def con_rule(model, m):
    return sum(mydata.a[i,m]*model.x[i] for i in mydata.N) \
                    >= mydata.b[m]
model.con = Constraint(model.M, rule=con_rule)
