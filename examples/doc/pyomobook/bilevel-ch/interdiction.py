# interdiction.py
from pyomo.environ import *
from pyomo.bilevel import *
from interdiction_data import A, budget

INDEX = list(A.keys())

M = ConcreteModel()
M.x = Var(INDEX, within=Binary)
M.budget = Constraint(expr=summation(M.x) <= budget)
M.sub = SubModel()
M.sub.f = Var()
M.sub.y = Var(INDEX, within=NonNegativeReals)

# Min/Max objectives
M.o = Objective(expr=M.sub.f, sense=minimize)
M.sub.o  = Objective(expr=M.sub.f, sense=maximize)

# Flow constraint
def flow_rule(M, n):
    return sum(M.y[i,n] for i in sequence(0,4) if (i,n) in A) == sum(M.y[n,j] for j in sequence(1,5) if (n,j) in A)
M.sub.flow = Constraint(sequence(1,4), rule=flow_rule)

# Source constraint
def s_rule(M):
    model = M.model()
    return model.sub.f <= sum(M.y[0,j] for j in sequence(1,4) if (0,j) in A)
M.sub.s = Constraint(rule=s_rule)

# Destination constraint
def t_rule(M):
    model = M.model()
    return model.sub.f <= sum(M.y[j,5] for j in sequence(1,4) if (j,5) in A)
M.sub.t = Constraint(rule=t_rule)

# Capacity constraint
def c_rule(M, i, j):
    model = M.model()
    return M.y[i,j] <= A[i,j]*(1-model.x[i,j])
M.sub.c = Constraint(INDEX, rule=c_rule)
