from pyomo.environ import *

model = ConcreteModel()

# @decl1:
model.A = Set()
# @:decl1
model.A.add(1,4,9)

# @decl2:
model.B = Set(initialize=[2,3,4])
model.C = Set(initialize=[(1,4),(9,16)])
# @:decl2

# @decl3:
model.D = Set(initialize=set([1,4,9]))
model.E = Set(initialize=(i for i in model.B if i%2 == 0))
# @:decl3

# @decl6:
F_init = {}
F_init[2] = [1,3,5]
F_init[3] = [2,4,6]
F_init[4] = [3,5,7]
model.F = Set([2,3,4],initialize=F_init)
# @:decl6

try:
# @decl7:
    model.G = Set()
    model.H = Set(model.G)
    model.H[2].add(4)
    model.H[4].add(16)
# @:decl7
except Exception:
    pass

# @decl4:
def I_init(model):
    return ((a,b) for a in model.A for b in model.B)
model.I = Set(initialize=I_init, dimen=2)
# @:decl4

# @decl8:
def J_init(model, i, j):
    return range(0,i*j)
model.J = Set(model.B,model.B, initialize=J_init)
# @:decl8

# @decl5:
def K_init(model, i):
    if i > 10:
        return Set.End
    return 2*i+1
model.K = Set(initialize=K_init)
# @:decl5

# @decl10:
def L_init(model, z):
    if z==6:
        return Set.End
    if z==1:
        return 1
    else:
        return model.L[z-1]*(z+1)
model.L = Set(ordered=True, initialize=L_init)
# @:decl10

# @decl10a:
@simple_set_rule
def LL_init(model, z):
    if z==6:
        return None
    if z==1:
        return 1
    else:
        return model.LL[z-1]*(z+1)
model.LL = Set(ordered=True, initialize=LL_init)
# @:decl10a

# @decl9:
def M_init(model, z, i, j):
    if z > 5:
        return Set.End
    return i*j+z
model.M = Set(model.B,model.B, initialize=M_init)
# @:decl9

# @decl11:
def N_init(model, z, i):
    if z==5:
        return Set.End
    if i == 5:
        return Set.End
    if z==1:
        if i==1:
            return 1
        else:
            return (z+1)
    return model.N[i][z-1]+z
model.N = Set(RangeSet(1,4), initialize=N_init, ordered=True)
# @:decl11

# @decl12:
model.P = Set(initialize=[1,2,3,5,7])
model.Q = Set(initialize=range(1,10),
             filter=lambda model, x:not x in model.P)
# @:decl12

# @decl13:
def FloorRoom_init(model, i):
    if i==1:
        return ['Lecture Hall']
    elif i==2:
        return ['Conference Room', 'Coffe Room']
    elif i==3:
        return range(301, 313)
model.FloorRoom = Set([1,2,3], initialize=FloorRoom_init)
# @:decl13

# @decl14:
def Floor_Room_init(model):
    retval = [(1, 'Lecture Hall'), (2, 'Conference Room')]
    retval = retval + [(2, 'Coffee Room')]
    retval = retval + [(3, i) for i in range(301, 313)]
    return retval
model.Floor_Room = Set(dimen=2, initialize=Floor_Room_init)
# @:decl14

# @decl20:
model.R = Set([1,2,3])
model.R[1] = [1]
model.R[2] = [1,2]
# @:decl20

#instance = model.create_instance()
model.pprint(verbose=True)
