from pyomo.environ import *
try:
    import numpy
    numpy_available=True
except ImportError:
    numpy_available=False

model = AbstractModel()

# @decl1:
model.Z = Param(initialize=32)
# @:decl1

# @decl2:
model.A = Set(initialize=[1,2,3], ordered=True)
y={}
y[1] = 10
y[2] = 20
y[3] = 30
model.Y = Param(model.A, initialize=y)
# @:decl2
model.del_component('Y')

# @decl3a:
def Y_init(model):
    return 2.0
model.Y = Param(initialize=Y_init)
# @:decl3a

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

# @decl6:
model.T = Param(model.A, model.B)
# @:decl6

# @decl5:
u={}
u[1,1] = 10
u[2,2] = 20
u[3,3] = 30
model.U = Param(model.A, model.A, initialize=u, default=0)
# @:decl5

if numpy_available:
# @decl7:
    vec = numpy.array([1,2,4,8])
    model.C = Set(initialize=range(len(vec)))
    def V_rule(model, i):
        return vec[i]
    model.V = Param(model.C, initialize=V_rule)
# @:decl7

instance = model.create_instance()
instance.pprint()
