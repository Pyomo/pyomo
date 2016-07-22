from pyomo.environ import *

model = ConcreteModel()

print('declscalar')
# @declscalar:
model.a = Objective()
# @:declscalar
model.display()

model = None
model = ConcreteModel()

print('declexprrule')
# @declexprrule:
model.x = Var([1,2], initialize=1.0)

model.b = Objective(expr=model.x[1] + 2*model.x[2])

def m_rule(model):
    expr = model.x[1]
    expr += 2*model.x[2]
    return expr
model.c = Objective(rule=m_rule)
# @:declexprrule
model.display()

model = None
model = ConcreteModel()

print('declmulti')
# @declmulti:
A = ['Q', 'R', 'S']
model.x = Var(A, initialize=1.0)
def d_rule(model, i):
    return model.x[i]**2
model.d = Objective(A, rule=d_rule)
# @:declmulti

print('declskip')
# @declskip:
def e_rule(model, i):
    if i == 'R':
        return Objective.Skip
    return model.x[i]**2
model.e = Objective(A, rule=e_rule)
# @:declskip
model.display()

model = None
model = ConcreteModel()

print('value')
# @value:
A = ['Q', 'R']
model.x = Var(A, initialize={'Q':1.5, 'R':2.5})
model.o = Objective(expr=model.x['Q'] + 2*model.x['R'])
print(model.o.expr)    # x[Q] + 2*x[R]
print(model.o.sense)   # minimize
print(value(model.o))  # 6.5
# @:value

model.display()
