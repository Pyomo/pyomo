# ConcHLinScript.py - Linear (H) as a script
from pyomo.environ import *

instance = ConcreteModel(name="Linear (H)")

A = ['I_C_Scoops', 'Peanuts']
h = {'I_C_Scoops': 1, 'Peanuts': 0.1}
d = {'I_C_Scoops': 5, 'Peanuts': 27}
c = {'I_C_Scoops': 3.14, 'Peanuts': 0.2718}
b = 12
u = {'I_C_Scoops': 100, 'Peanuts': 40.6}

def x_bounds(m, i):
    return (0,u[i])

instance.x = Var(A, bounds=x_bounds)

def obj_rule(instance):
    return sum(h[i]*(1 - u[i]/d[i]**2) * instance.x[i] for i in A)

instance.z = Objective(rule=obj_rule, sense=maximize)

instance.budgetconstr = \
     Constraint(expr = sum(c[i] * instance.x[i] for i in A) <= b)

# @tail:
opt = SolverFactory('glpk')

results = opt.solve(instance) # solves and updates instance

instance.display()
# @:tail
