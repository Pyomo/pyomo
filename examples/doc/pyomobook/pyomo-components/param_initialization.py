from pyomo.environ import *

model = AbstractModel()
model.A = Set(initialize=[1,2,3])

# @decl1:
model.Z = Param(initialize=32)
# @:decl1

# @decl3b:
def X_init(model, i, j):
    return i*j
model.X = Param(model.A, model.A, initialize=X_init)
# @:decl3b

# @decl3c:
def XX_init(model, i, j):
    if i==1 or j==1:
        return i*j
    return i*j + model.XX[i-1,j-1]
model.XX = Param(model.A, model.A, initialize=XX_init)
# @:decl3c

# @decl4:
model.B = Set(initialize=[1,2,3])
w={}
w[1] = 10
w[3] = 30
model.W = Param(model.B, initialize=w)
# @:decl4

# @decl5:
u={}
u[1,1] = 10
u[2,2] = 20
u[3,3] = 30
model.U = Param(model.A, model.A, initialize=u, default=0)
# @:decl5

# @decl6:
model.T = Param(model.A, model.B)
# @:decl6

instance = model.create_instance()
instance.pprint()


# --------------------------------------------------
# @special1:
model = ConcreteModel()
model.p = Param([1,2,3], initialize={1:1.42, 3:3.14})
model.q = Param([1,2,3], initialize={1:1.42, 3:3.14}, default=0)

# Demonstrating the len() function
print(len(model.p))                 # 2
print(len(model.q))                 # 3

# Demonstrating the 'in' operator (checks against component keys)
print(2 in model.p)                 # False
print(2 in model.q)                 # True

# Demonstrating iteration over component keys
print([key for key in model.p])     # [1,3]
print([key for key in model.q])     # [1,2,3]
# @:special1
