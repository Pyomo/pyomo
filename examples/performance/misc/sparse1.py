#
# This is a performance test that we cannot easily execute right now
#
from pyomo.environ import *


def f(N):
    M = ConcreteModel()
    M.A = Set(initialize=range(N))
    M.x = Var()
    M.o = Objective(expr=M.x)
    def rule(m, i):
        if i == 3 or i == 5:
            return M.x >= i
        return Constraint.Skip
    M.c = Constraint(M.A, rule=rule)
    return M

#
# Generation of this model is slow because set M.A is big
#
model = f(1000000)
