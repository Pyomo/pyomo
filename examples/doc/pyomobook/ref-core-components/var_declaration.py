from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.x = Var()
# @:decl1

# @decl2:
model.A = Set(initialize=[1,2,3])
model.y = Var(within=model.A)
model.r = Var(domain=Reals)
model.w = Var(within=Boolean)
# @:decl2

# @decl2a:
def s_domain(model, i):
    return IntegerInterval(bounds=(i,i+1))
model.s = Var(model.A, domain=s_domain)
# @:decl2a

# @decl3:
B = [1.5, 2.5, 3.5]
model.u = Var(B)
model.C = Set()
model.t = Var(B, model.C)
# @:decl3

# @decl4:
model.a = Var(bounds=(0.0,None))

lower = {1:2, 2:4, 3:6}
upper = {1:3, 2:4, 3:7}
def f(model, i):
    return (lower[i], upper[i])
model.b = Var(model.A, bounds=f)
# @:decl4

# @decl5:
model.za = Var(initialize=9, within=NonNegativeReals)
model.zb = Var(model.A, initialize={1:1, 2:4, 3:9}, dense=True)
model.zc = Var(model.A, initialize=2)
# @:decl5

# @decl6:
def g(model, i):
    return 3*i
model.m = Var(model.A, initialize=g)
# @:decl6

model = model.create_instance('var_declaration.dat')
# @decl13:
model.Za = Var(within=NonNegativeReals)
model.Zb = Var(model.A)
model.Zc = Var(model.A)

Za_values = {None: 9}
model.Za.set_values(Za_values)
Zb_values = {1:1, 2:4, 3:9}
model.Zb.set_values(Zb_values)
Zc_values = {1:2, 2:2, 3:2}
model.Zc.set_values(Zc_values)
# @:decl13


# @decl12:
model.v = VarList(within=Integers)
v = model.v.add()
model.v[1].setlb(1)
model.v[1].setub(3)
v = model.v.add()
v.setlb(5)
v.setub(7)
# @:decl12

# @decl7a:
model.za = 8.5
print(model.za.value)        # 8.5
# @:decl7a

# @decl7:
model.za = 8.5
model.zb[2] = 7
# @:decl7

# @decl9:
print(value(model.za))       # 8.5
print(value(model.zb[2]))    # 7
# @:decl9

# @decl10:
print(len(model.za))         # 1
print(len(model.zb))         # 3
# @:decl10

# @decl11:
print(model.zb[2].value)     # 7
print(model.za.lb)           # 0
print(model.za.ub)           # None
print(model.zb[2].fixed)     # False
# @:decl11

for i in model.b.index_set():
    model.b[i]
for i in model.m.index_set():
    model.m[i]
for i in model.r.index_set():
    model.r[i]
for i in model.s.index_set():
    model.s[i]
for i in model.t.index_set():
    model.t[i]
for i in model.u.index_set():
    model.u[i]
for i in model.zb.index_set():
    model.zb[i]
for i in model.zc.index_set():
    model.zc[i]

model.pprint()
