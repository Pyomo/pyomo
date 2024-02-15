#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.ConcreteModel()
# @len:
model.A = pyo.Set(initialize=[1, 2, 3])

print(len(model.A))  # 3
# @:len

model = pyo.ConcreteModel()
# @data:
model.A = pyo.Set(initialize=[1, 2, 3])
model.B = pyo.Set(initialize=[3, 2, 1], ordered=True)
model.C = pyo.Set(model.A, initialize={1: [1], 2: [1, 2]})

print(type(model.A.data()) is tuple)  # True
print(type(model.B.data()) is tuple)  # True
print(type(model.C.data()) is dict)  # True
print(sorted(model.A.data()))  # [1, 2, 3]
for index in sorted(model.C.data().keys()):
    print(sorted(model.C.data()[index]))
# [1]
# [1, 2]
# @:data

model = pyo.ConcreteModel()
# @special:
model.A = pyo.Set(initialize=[1, 2, 3])

# Test if an element is in the set
print(1 in model.A)  # True

# Test if sets are equal
print([1, 2] == model.A)  # False

# Test if sets are not equal
print([1, 2] != model.A)  # True

# Test if a set is a subset of or equal to the set
print([1, 2] <= model.A)  # True

# Test if a set is a subset of the set
print([1, 2] < model.A)  # True

# Test if a set is a superset of the set
print([1, 2, 3] > model.A)  # False

# Test if a set is a superset of or equal to the set
print([1, 2, 3] >= model.A)  # True
# @:special

model = pyo.ConcreteModel()
# @iter:
model.A = pyo.Set(initialize=[1, 2, 3])
model.C = pyo.Set(model.A, initialize={1: [1], 2: [1, 2]})

print(sorted(e for e in model.A))  # [1, 2, 3]
for index in model.C:
    print(sorted(e for e in model.C[index]))
# [1]
# [1, 2]
# @:iter

model = pyo.ConcreteModel()
# @ordered:
model.A = pyo.Set(initialize=[3, 2, 1], ordered=True)

print(model.A.first())  # 3
print(model.A.last())  # 1
print(model.A.next(2))  # 1
print(model.A.prev(2))  # 3
print(model.A.nextw(1))  # 3
print(model.A.prevw(3))  # 1
# @:ordered

model = pyo.ConcreteModel()
# @ordered2:
model.A = pyo.Set(initialize=[3, 2, 1], ordered=True)

print(model.A.ord(3))  # 1
print(model.A.ord(1))  # 3
print(model.A[1])  # 3
print(model.A[3])  # 1
# @:ordered2
