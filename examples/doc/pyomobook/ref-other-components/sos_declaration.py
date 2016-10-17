from pyomo.environ import *


print("*"*5 + " decl1 ")
model = ConcreteModel()
# @decl1:
model.x = Var([1,2,3,4,5], dense=True)
model.c = SOSConstraint(var=model.x, sos=1)
# @:decl1
model.pprint()

print("*"*5 + " decl2 ")
model = ConcreteModel()
# @decl2:
model.x = Var([1,2,3,4,5], dense=True)
model.c = SOSConstraint(var=model.x, level=1)
# @:decl2
model.pprint()

print("*"*5 + " decl3 ")
model = ConcreteModel()
# @decl3:
model.A = Set(initialize=[1,2,3,4,5], ordered=True)
model.x = Var(model.A, dense=True)
model.c = SOSConstraint(var=model.x, sos=2)
# @:decl3
model.pprint()

print("*"*5 + " decl4 ")
model = ConcreteModel()
# @decl4:
model.x = Var([1,2,3,4,5])
model.c = SOSConstraint(var=model.x, index=[1,3,5], sos=1)
# @:decl4
model.pprint()

print("*"*5 + " decl5 ")
model = ConcreteModel()
# @decl5:
model.x = Var([1,2,3,4,5])
model.c = SOSConstraint(var=model.x, index=[1,3,5], sos=2)
# @:decl5
model.pprint()

print("*"*5 + " decl6 ")
model = ConcreteModel()
# @decl6:
w = {1:1, 2:2, 3:3, 4:4, 5:5}
model.x = Var([1,2,3,4,5])
model.c = SOSConstraint(var=model.x, weights=w, sos=1)
# @:decl6
model.pprint()

print("*"*5 + " decl7 ")
model = ConcreteModel()
# @decl7:
model.x = Var([1,2,3,4,5], dense=True)

def c1_rule(model):
    return list(model.x.itervalues())
model.c1 = SOSConstraint(rule=c1_rule, sos=1)

model.c2 = SOSConstraint(var=model.x, sos=1)
# @:decl7
model.pprint()

print("*"*5 + " decl8 ")
model = ConcreteModel()
# @decl8:
model.x = Var([1,2,3,4,5], dense=True)

def c_rule(model, i):
    return [model.x[j] for j in model.x if j%2 == i]
model.c = SOSConstraint([0,1], rule=c_rule, sos=1)
# @:decl8
model.pprint()

print("*"*5 + " decl9 ")
model = ConcreteModel()
# @decl9:
model.x = Var([1,2,3,4,5], dense=True)

def c_rule(model, i):
    if i%2 == 0:
        return SOSConstraint.Skip
    return [model.x[j] for j in model.x if j>=i]
model.c = SOSConstraint([1,2,3], rule=c_rule, sos=1)
# @:decl9
model.pprint()

print("*"*5 + " decl10 ")
model = ConcreteModel()
# @decl10:
model.x = Var([1,2,3,4,5], dense=True)

def c_rule(model, i,j):
    return [model.x[k] for k in model.x if k>=i+j]
model.c = SOSConstraint([1,2], [1,2], rule=c_rule, sos=1)
# @:decl10
model.pprint()

print("*"*5 + " decl11 ")
model = ConcreteModel()
# @decl11:
model.x = Var([1,2,3,4,5], dense=True)

def c_rule(model, i):
    v = []
    w = []
    for j in model.x:
        v.append(model.x[j])
        w.append(i*10+j)
    return v,w
model.c = SOSConstraint([0,1], rule=c_rule, sos=1)
# @:decl11
model.pprint()

print("*"*5 + " decl12 ")
model = ConcreteModel()
# @decl12:
model.x = Var([1,2,3,4,5], dense=True)

def c_rule(model):
    v = []
    for j in sorted(model.x.iterkeys(), reverse=True):
        v.append(model.x[j])
    return v
model.c = SOSConstraint(rule=c_rule, sos=2)
# @:decl12
model.pprint()

print("*"*5 + " decl13 ")
model = ConcreteModel()
# @decl13:
model.x = Var([1,2,3,4,5], dense=True)
model.y = Var([1,2,3,4,5], dense=True)

def c_rule(model):
    return list(model.x.itervalues()) + \
           list(model.y.itervalues())
model.c = SOSConstraint(rule=c_rule, sos=1)
# @:decl13
model.pprint()

