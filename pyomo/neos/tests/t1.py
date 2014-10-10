
import coopr.environ
from coopr.pyomo import *

M = ConcreteModel()
M.x = Var(bounds=(0,1))
M.o = Objective(expr=2*M.x, sense=maximize)
M.c = Constraint(expr=M.x <= 0.5)

model = M

