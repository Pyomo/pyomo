# interdiction.py

from pyomo.environ import *
from pyomo.bilevel import *

nodes = range(6)
A = {(0,1):2, (0,2):6, (0,3):4, (0,4):2, 
     (1,5):3, 
     (2,1):2, (2,3):2,
     (3,4):2,
     (4,5):1,
     (3,5):5}


M = ConcreteModel()

M.f = Var()
M.x = Var(A.keys(), within=Binary)
M.y = Var(A.keys(), within=NonNegativeReals)

M.o  = Objective(expr=M.f, sense=maximize)

# Flow constraint
def flow_rule(M, n):
    return sum(M.y[i,n] for i in sequence(0,4) if (i,n) in A) == \
           sum(M.y[n,j] for j in sequence(1,5) if (n,j) in A)
M.flow = Constraint(sequence(1,4), rule=flow_rule)

# Source constraint
def s_rule(M):
    return M.f <= sum(M.y[0,j] for j in sequence(1,4) if (0,j) in A)
M.s = Constraint(rule=s_rule)

# Destination constraint
def t_rule(M):
    return M.f <= sum(M.y[j,5] for j in sequence(1,4) if (j,5) in A)
M.t = Constraint(rule=t_rule)

# Capacity constraint
def c_rule(M, i, j):
    return M.y[i,j] <= A[i,j]
M.c = Constraint(A.keys(), rule=c_rule)

model = M
