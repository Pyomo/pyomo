from pyomo.environ import *

model = ConcreteModel()
# @len:
model.A = Set(initialize=[1,2,3])

print(len(model.A))     # 3
# @:len

model = ConcreteModel()
# @data:
model.A = Set(initialize=[1,2,3])
model.B = Set(initialize=[3, 2, 1], ordered=True)
model.C = Set(model.A, initialize={1:[1], 2:[1,2]})

print(type(model.A.data()))     # set
print(type(model.B.data()))     # set
print(type(model.C.data()))     # dict
print(sorted(model.A.data()))   # [1,2,3]
# [1]
# [1,2]
for index in sorted(model.C.data().keys()):
  print(sorted(model.C.data()[index]))
# @:data

model = ConcreteModel()
# @special:
model.A = Set(initialize=[1,2,3])

# Test is an element is in the set
print(1 in model.A)         # True

# Test if sets are equal
print([1,2] == model.A)     # False

# Test if sets are not equal
print([1,2] != model.A)     # True

# Test if a set is a subset or equal of the set
print([1,2] <= model.A)     # True

# Test if a set is a subset of the set
print([1,2] < model.A)     # True

# Test if a set is a superset of the set
print([1,2,3] > model.A)   # False

# Test if a set is a superset of the set
print([1,2,3] >= model.A)  # True
# @:special

model = ConcreteModel()
# @iter:
model.A = Set(initialize=[1,2,3])
model.C = Set(model.A, initialize={1:[1], 2:[1,2]})

print(sorted(e for e in model.A))   # [1,2,3]
# [1]
# [1,2]
for index in model.C:
    print(sorted(e for e in model.C[index]))
# @:iter

model = ConcreteModel()
# @ordered:
model.A = Set(initialize=[3,2,1], ordered=True)

print(model.A.first())      # 3
print(model.A.last())       # 1
print(model.A.next(2))      # 1
print(model.A.prev(2))      # 3
print(model.A.nextw(1))     # 3
print(model.A.prevw(3))     # 1
# @:ordered

model = ConcreteModel()
# @ordered2:
model.A = Set(initialize=[3,2,1], ordered=True)

print(model.A.ord(3))       # 1
print(model.A.ord(1))       # 3
print(model.A[1])           # 3
print(model.A[3])           # 1
# @:ordered2

model = ConcreteModel()
# @operators:
model.A = Set(initialize=[1,2,3])
model.B = Set(initialize=[2,3,4])

model.M = Set(initialize=[0,1,2])
# m = M | A | B
model.m = model.M.union(model.A, model.B)

model.N = Set(initialize=[0,1,4])
# n = N - A - B
model.n = model.N.difference(model.A, model.B)

model.O = Set(initialize=[0,1,2])
# o = O & A & B
model.o = model.O.intersection(model.A, model.B)

model.P = Set(initialize=[0,1,2])
# p = P ^ A ^ B
model.p = model.P.symmetric_difference(model.A, model.B)

model.Q = Set(initialize=[0,1,2])
# q = Q * A * B
model.q = model.Q.cross(model.A, model.B)
# @:operators
print(sorted(model.m.data()))
print(sorted(model.n.data()))
print(sorted(model.o.data()))
print(sorted(model.p.data()))
print(sorted(model.q.data()))

model = ConcreteModel()
# @setof1:
a = [1,2,3]

model.A = Set(initialize=a)     # copy data
model.B = SetOf(initialize=a)   # do not copy

print(sorted(model.A.data()))   # [1,2,3]
print(sorted(model.B.data()))   # [1,2,3]

a[1] = 4

print(sorted(model.A.data()))   # [1,2,3]
print(sorted(model.B.data()))   # [1,4,3]
# @:setof1

model = ConcreteModel()
# @setof2:
a = [1,2,3]

model.B = SetOf(initialize=a)   # do not copy
model.C = SetOf(a)              # do not copy
# @:setof2
