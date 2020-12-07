from pyomo.core import *

INDEX_SET1 = [1,2]
INDEX_SET2 = [1,2,3]
PIECEWISE_PTS = dict([((t1,t2),[1.0,2.0,3.0]) for t1 in INDEX_SET1 for t2 in INDEX_SET2])
F = lambda x: x**2


model = ConcreteModel()

def SOS_indices_init(model,t1,t2):
    return [(t1,t2,i) for i in xrange(1,len(PIECEWISE_PTS[t1,t2]))]
def ub_indices_init(model):
    return [(t1,t2,i) for t1 in INDEX_SET1 for t2 in INDEX_SET2 for i in xrange(1,len(PIECEWISE_PTS[t1,t2]))]

model.SOS_indices = Set(INDEX_SET1,INDEX_SET2,dimen=3, ordered=True, initialize=SOS_indices_init)
model.ub_indices = Set(ordered=True, dimen=3,initialize=ub_indices_init)

model.x = Var(INDEX_SET1,INDEX_SET2)
model.Fx = Var(INDEX_SET1,INDEX_SET2)
#Add SOS1 variable to model
model.y = Var(model.ub_indices,within=NonNegativeReals)

def constraint1_rule(model,t1,t2,i):
    return model.Fx[t1,t2] - (F(PIECEWISE_PTS[t1,t2][i-1]) + ((F(PIECEWISE_PTS[t1,t2][i])-F(PIECEWISE_PTS[t1,t2][i-1]))/(PIECEWISE_PTS[t1,t2][i]-PIECEWISE_PTS[t1,t2][i-1]))*(model.x[t1,t2]-PIECEWISE_PTS[t1,t2][i-1])) <= 25.0*(1-model.y[t1,t2,i])
def constraint2_rule(model,t1,t2):
    return sum(model.y[t1,t2,j] for j in xrange(1,len(PIECEWISE_PTS[t1,t2]))) == 1

model.obj = Objective(expr=sum_product(model.Fx), sense=maximize)
model.constraint1 = Constraint(model.ub_indices,rule=constraint1_rule)
model.constraint2 = Constraint(INDEX_SET1,INDEX_SET2,rule=constraint2_rule)
model.SOS_set_constraint = SOSConstraint(INDEX_SET1,INDEX_SET2, var=model.y, index=model.SOS_indices, sos=1)

#Fix the answer for testing purposes
model.set_answer_constraint1 = Constraint(expr= model.x[1,1] == 2.5)
model.set_answer_constraint2 = Constraint(expr= model.x[2,1] == 1.5)
model.set_answer_constraint3 = Constraint(expr= model.x[1,2] == 2.5)
model.set_answer_constraint4 = Constraint(expr= model.x[2,2] == 1.5)
model.set_answer_constraint5 = Constraint(expr= model.x[1,3] == 2.5)
model.set_answer_constraint6 = Constraint(expr= model.x[2,3] == 1.5)
