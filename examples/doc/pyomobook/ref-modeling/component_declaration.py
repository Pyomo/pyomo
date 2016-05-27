from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.A = Set(initialize=[1,2,3])
model.B = Set(initialize=[(1,1),(2,2),(3,3)])

model.z = Var(model.B, initialize=1.0)
model.y = Var(model.A, model.B, initialize=1.0)
# @:decl1

# @decl2:
model.x = Var(['a','b'], model.B, initialize=1.0)
# @:decl2


# @decl3a:
def C_index(model):
    for k in model.A:
        if k % 2 == 0:
            yield k
model.w = Var(C_index, model.B, initialize=1.0)
# @:decl3a

# @decl3b:
def D_index(model):
    for i in model.B:
        k = sum(i)
        if k % 2 == 0:
            yield (k,k)
D_index.dimen=2
model.v = Var(D_index, model.B, initialize=1.0)
# @:decl3b


instance = model.create_instance()
for i in instance.z.index_set():
    instance.z[i]
for i in instance.y.index_set():
    instance.y[i]
for i in instance.x.index_set():
    instance.x[i]
for i in instance.v.index_set():
    instance.v[i]
for i in instance.w.index_set():
    instance.w[i]
instance.pprint()
