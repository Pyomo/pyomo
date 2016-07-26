from pyomo.environ import *

model = AbstractModel()

# @decl1:
model.Z = Param()
# @:decl1

# @decl2:
model.V = Set(initialize=[1,2,3])
model.Y = Param(within=model.V)
model.X = Param(within=Reals)
model.W = Param(within=Boolean)
# @:decl2


# @decl3:
model.A = Set(initialize=[1,2,3])
model.B = Set()
model.U = Param(model.B)
model.C = Set()
model.T = Param(model.A, model.C)
# @:decl3

instance = model.create_instance('param_declaration.dat')
instance.pprint()
