from pyomo.core import *

INDEX_SET = [1,2]
PIECEWISE_PTS = {1:[1,2,3], 2:[1,2,3]}
F = lambda x: x**2

model = ConcreteModel()

def SOS_indices_init(model,t):
    return [(t,i) for i in xrange(1,len(PIECEWISE_PTS[t]))]
def ub_indices_init(model):
    return [(t,i) for t in INDEX_SET for i in xrange(1,len(PIECEWISE_PTS[t]))]

model.SOS_indices = Set(INDEX_SET,dimen=2, ordered=True, initialize=SOS_indices_init)
model.ub_indices = Set(ordered=True, dimen=2,initialize=ub_indices_init)

model.x = Var(INDEX_SET)
model.Fx = Var(INDEX_SET)
#Add SOS1 variable to model
model.y = Var(model.ub_indices,within=NonNegativeReals)

def constraint1_rule(model,t,i):
    return model.Fx[t] - (F(PIECEWISE_PTS[t][i-1]) + ((F(PIECEWISE_PTS[t][i])-F(PIECEWISE_PTS[t][i-1]))/(PIECEWISE_PTS[t][i]-PIECEWISE_PTS[t][i-1]))*(model.x[t]-PIECEWISE_PTS[t][i-1])) <= 25.0*(1-model.y[t,i])
def constraint2_rule(model,t):
    return sum(model.y[t,j] for j in xrange(1,len(PIECEWISE_PTS[t]))) == 1

model.obj = Objective(expr=sum_product(model.Fx), sense=maximize)
model.constraint1 = Constraint(model.ub_indices,rule=constraint1_rule)
model.constraint2 = Constraint(INDEX_SET,rule=constraint2_rule)
model.SOS_set_constraint = SOSConstraint(INDEX_SET, var=model.y, index=model.SOS_indices, sos=1)

#Fix the answer for testing purposes
model.set_answer_constraint1 = Constraint(expr= model.x[1] == 2.5)
model.set_answer_constraint2 = Constraint(expr= model.x[2] == 1.5)
