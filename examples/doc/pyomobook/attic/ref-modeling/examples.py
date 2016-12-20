from pyomo.environ import *

print("indexed1")
# --------------------------------------------------
# @indexed1:
model = AbstractModel()
model.A = Set()
model.x = Var()
model.o = Objective(expr=model.x)
model.c = Constraint(expr = model.x >= 0)
def d_rule(model, a):
    return a*model.x <= 0
model.d = Constraint(model.A, rule=d_rule)
# @:indexed1
data = DataPortal()
data['A'] = {None: set([1,2,3])}
instance = model.create_instance(data)
instance.pprint()

print("active1")
# --------------------------------------------------
# @active1:
model = ConcreteModel()
model.x = Var([1,2])
model.c = Constraint(expr= model.x[1] >= 0)
model.d = Constraint(expr= model.x[2] >= 0)
def e_rule(model, i):
    return i*model.x[1] + model.x[2] == 1
model.e = Constraint([1,2], rule=e_rule)

# Deactivate a single constraint
print(model.d.active)          # True
model.d.deactivate()
print(model.d.active)          # False

# Deactivate a collection of constraints
print(model.e.active)          # True
print(model.e[1].active)       # True
model.e.deactivate()
print(model.e[1].active)       # False
print(model.e.active)          # False

# Reactivate a collection of constraints
model.e.activate()
print(model.e[1].active)       # True

# Deactivate a single constraint in a collection
model.e[1].deactivate()
print(model.e.active)          # True
print(model.e[1].active)       # False
# @:active1

print("blocks1")
# --------------------------------------------------
# @blocks1:
model = ConcreteModel()
model.x = Var()
model.b = Block()
model.b.x = Var()
model.b.y = Var()
# @:blocks1
model.pprint()

print("nested1")
# --------------------------------------------------
# @nested1:
model = ConcreteModel()
model.b = Block()
model.b.x = Var()

print(model.b.x.local_name)   # 'x'
print(model.b.x.name)         # 'b.x'
# @:nested1

print("special1")
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

# Demonstrating the '[]' operator
print(model.p[1])                   # 1.42
print(model.q[1])                   # 1.42
# @:special1

print("special2")
# --------------------------------------------------
# @special2:
model = ConcreteModel()
model.p = Var([1,2,3], initialize={1:1.42, 3:3.14})
model.q = Var([1,2,3], initialize={1:1.42, 2: 2.5, 3:3.14})

# Demonstrating the len() function
print(len(model.p))                 # 2
print(len(model.q))                 # 3

# Demonstrating the 'in' operator (checks against component keys)
print(2 in model.p)                 # False
print(2 in model.q)                 # True

# Demonstrating iteration over component keys
print([key for key in model.p])     # [1,3]
print([key for key in model.q])     # [1,2,3]

# Demonstrating the '[]' operator
print(model.p[1])                   # 1.42
print(model.q[1])                   # 1.42
# @:special2

print("indexed2")
# --------------------------------------------------
# @indexed2:
model = ConcreteModel()
model.x = Var([1,2,3], initialize={1:1.42, 3:3.14})

# Demonstrating the len() function
print(len(model.x))                 # 3

# Demonstrating the 'in' operator (checks against component keys)
print(2 in model.x)                 # True
print(4 in model.x)                 # False

# Demonstrating iteration over component keys
print([key for key in model.x])     # [1,2,3]

# Demonstrating the '[]' operator
print(value(model.x[1]))            # 1.42

# Demonstrating the keys() function
print(list(model.x.keys()))         # [1,2,3]
# @:indexed2

# Demonstrating the items() function
#print(list(model.x.items()))        # [(1,1.42), (2,None), (3,3.14)]

# Demonstrating the values() function
#print(list(model.x.values()))       # [1.42,None,3.14]

print("indexed3")
# --------------------------------------------------
# @indexed3:
model = ConcreteModel()
model.p = Param([1,2,3], initialize={1:1.42, 3:3.14})
model.q = Param([1,2,3], initialize={1:1.42, 3:3.14}, default=0)

# Demonstrating the keys() function
print(list(model.p.iterkeys()))     # [1,3]
print(list(model.q.iterkeys()))     # [1,2,3]

# Demonstrating the items() function
print(list(model.p.iteritems()))    # [(1,1.42), (3,3.14)]
print(list(model.q.iteritems()))    # [(1,1.42), (2,0), (3,3.14)]

# Demonstrating the values() function
print(list(model.p.itervalues()))   # [1.42,3.14]
print(list(model.q.itervalues()))   # [1.42,0,3.14]
# @:indexed3

print("indexed4")
# --------------------------------------------------
# @indexed4:
model = ConcreteModel()
model.p = Param([1,2])
model.q = Param([1,2], [3,4])

# The index set of p is [1,2]
print(list(model.p.index_set()))
# The index set of q is [(1,3), (1,4), (2,3), (2,4)]
print(list(model.q.index_set()))

# Demonstrating the dim() function
print(model.p.dim())        # 1
print(model.q.dim())        # 2
# @:indexed4

print("numvalue1")
# --------------------------------------------------
# @numvalue1:
model = ConcreteModel()
# A single parameter is a subclass of NumericValue
model.p = Param(initialize=3)

model.p + 2             # Calls __add__
model.p - 2             # Calls __sub__
model.p * 2             # Calls __mul__
model.p / 2             # Calls __div__ or __truediv__
model.p ** 2            # Calls __pow__

2 + model.p             # Calls __radd__
2 - model.p             # Calls __rsub__
2 * model.p             # Calls __rmul__
2 / model.p             # Calls __rdiv__ or __rtruediv__
2 ** model.p            # Calls __rpow__

e = model.p
e += 2                  # Calls __iadd__
e -= 2                  # Calls __isub__
e *= 2                  # Calls __imul__
e /= 2                  # Calls __idiv__ or __itruediv__
e **= 2                 # Calls __ipow__

+ model.p               # Calls __pos__
- model.p               # Calls __neg__
# @:numvalue1

print("numvalue2")
# --------------------------------------------------
# @numvalue2:
model = ConcreteModel()
# A single parameter is a subclass of NumericConstant
model.p = Param(initialize=-3)

abs(model.p)            # Calls __abs__

# Special methods for NumericConstant objects
if model.p:             # Calls __nonzero__ in test
    pass
model.p()               # Calls __call__
# @:numvalue2

print("numvalue3")
# --------------------------------------------------
# --------------------------------------------------
# @numvalue3:
model = ConcreteModel()
model.p = Param()
model.x = Var()

print(model.p.polynomial_degree())      # 0
print(model.x.polynomial_degree())      # 1
e = model.x*model.p
print(e.polynomial_degree())            # 1
e = model.x*model.x
print(e.polynomial_degree())            # 2
e = model.x ** 2
print(e.polynomial_degree())            # 2
e = sin(model.x)
print(e.polynomial_degree())            # None
# @:numvalue3
