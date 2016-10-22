from pyomo.environ import *

model = ConcreteModel()

# @initscalarvar:
model.x = Var(initialize=3.14)
# @:initscalarvar

# for testing
print('3.14 =', value(model.x)) # 3.14

model = None
model = ConcreteModel()

# @dictruleinit:
model.A = Set(initialize=[1,2,3])
model.x = Var(model.A, initialize=3.14)
model.y = Var(model.A, initialize={1:1.5, 2:4.5, 3:5.5})
def z_init_rule(m, i):
    return float(i) + 0.5
model.z = Var(model.A, initialize=z_init_rule)
# @:dictruleinit

# for testing
print('[3.14, 3.14, 3.14] =', [value(model.x[i]) for i in model.x])
print('[1.5, 4.5, 5.5] =', [value(model.y[i]) for i in model.y])
print('[1.5, 2.5, 3.5] =', [value(model.z[i]) for i in model.x])

model = None
model = ConcreteModel()

# @declscalar:
model.x = Var()
# @:declscalar

# for testing
print("x =", model.x)

model = None
model = ConcreteModel()

# @decl3:
B = [1.5, 2.5, 3.5]
model.u = Var(B)
model.C = Set()
model.t = Var(B, model.C)
# @:decl3

# for testing
print(B)
print([k for k in model.u.keys()])

model = None
model = ConcreteModel()

# @domaindecl:
model.A = Set(initialize=[1,2,3])
model.y = Var(within=model.A)
model.r = Var(domain=Reals)
model.w = Var(within=Boolean)
# @:domaindecl

# for testing
model.A.pprint()
print(model.y.domain)
print(model.r.domain)
print(model.w.domain)

model = None
model = ConcreteModel()

# @domaindeclrule:
model.A = Set(initialize=[1,2,3])
def s_domain(model, i):
    return IntegerInterval(bounds=(i,i+1))
model.s = Var(model.A, domain=s_domain)
# @:domaindeclrule

# for testing
model.A.pprint()
model.s.pprint()

model = None
model = ConcreteModel()

# @declbounds:
model.A = Set(initialize=[1,2,3])
model.a = Var(bounds=(0.0,None))

lower = {1:2.5, 2:4.5, 3:6.5}
upper = {1:3.5, 2:4.5, 3:7.5}
def f(model, i):
    return (lower[i], upper[i])
model.b = Var(model.A, bounds=f)
# @:declbounds

# for testing
model.A.pprint()
print(model.a.lb)
print(model.a.ub)
print(model.b[3].lb)
print(model.b[3].ub)

model = None
model = ConcreteModel()

# @declinit:
model.A = Set(initialize=[1,2,3])
model.za = Var(initialize=9.5, within=NonNegativeReals)
model.zb = Var(model.A, initialize={1:1.5, 2:4.5, 3:5.5})
model.zc = Var(model.A, initialize=2.1)

print(value(model.za))    # 9.5
print(value(model.zb[3])) # 5.5
print(value(model.zc[3])) # 2.1
# @:declinit

model = None
model = ConcreteModel()

# @declinitrule:
model.A = Set(initialize=[1,2,3])
def g(model, i):
    return 3*i
model.m = Var(model.A, initialize=g)

print(value(model.m[1])) # 3
print(value(model.m[3])) # 9
# @:declinitrule

model = None
model = ConcreteModel()

print("varattrib")
# @varattrib:
model.A = Set(initialize=[1,2,3])
model.za = Var(initialize=9.5, within=NonNegativeReals)
model.zb = Var(model.A, initialize={1:1.5, 2:4.5, 3:5.5})
model.zc = Var(model.A, initialize=2.1)

print(value(model.zb[2]))     # 4.5
print(model.za.lb)           # 0
print(model.za.ub)           # None

model.za = 8.5
model.zb[2] = 7.5

print(value(model.za)) # 8.5

Zc_values = {1:2.5, 2:2.5, 3:2.5}
model.zc.set_values(Zc_values)
print(value(model.zc[2])) # 2.5

model.zb.fix(3.0)
print(model.zb[1].fixed)     # True
print(model.zb[2].fixed)     # True
print(value(model.zb[1]))    # 3.0
print(value(model.zb[2]))    # 3.0
model.zc[2].fix(3.0)
print(model.zc[1].fixed)     # False
print(model.zc[2].fixed)     # True
print(value(model.zc[1]))    # 2.5
print(value(model.zc[2]))    # 3.0

# @:varattrib

