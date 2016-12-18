# interdiction.py

from pyomo.environ import *
from pyomo.bilevel import *
from interdiction_data import A, budget

M = ConcreteModel()
M.f = Var()
M.x = Var(A.keys(), within=Binary)
M.o = Objective(expr=M.f, sense=minimize)
M.budget = Constraint(expr=summation(M.x) <= budget)

M.sub = SubModel(fixed=M.x)
M.sub.o  = Objective(expr=M.f, sense=maximize)
M.sub.y = Var(A.keys(), within=NonNegativeReals)

# Flow constraint
def flow_rule(M, n):
    return sum(M.y[i,n] for i in sequence(0,4) if (i,n) in A) == \
           sum(M.y[n,j] for j in sequence(1,5) if (n,j) in A)
M.sub.flow = Constraint(sequence(1,4), rule=flow_rule)

# Source constraint
def s_rule(M):
    model = M.model()
    return model.f <= sum(M.y[0,j] for j in sequence(1,4) if (0,j) in A)
M.sub.s = Constraint(rule=s_rule)

# Destination constraint
def t_rule(M):
    model = M.model()
    return model.f <= sum(M.y[j,5] for j in sequence(1,4) if (j,5) in A)
M.sub.t = Constraint(rule=t_rule)

# Capacity constraint
def c_rule(M, i, j):
    model = M.model()
    return M.y[i,j] <= A[i,j]*(1-model.x[i,j])
M.sub.c = Constraint(A.keys(), rule=c_rule)

model = M
