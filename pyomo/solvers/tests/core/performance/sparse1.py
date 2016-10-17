#
# This is a performance test that we cannot easily execute right now
#
import pyomo.environ
from pyomo.core import *


def f(n):
    M = ConcreteModel()
    M.A = Set(initialize=range(n))
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
model = f(10**7)
#
# But writing the NL file should be fast because constraint M.c is small
#
M.write(filename='foo.nl', format=ProblemFormat.nl)
