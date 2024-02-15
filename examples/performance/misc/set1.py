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

from pyomo.environ import *

model = ConcreteModel()

model.d = Param(default=10)


def A_rule(model):
    return range(0, value(model.d))


model.A = Set()


def B_rule(model):
    return range(1, value(model.d) + 1)


model.B = Set()

if 1 > 0:
    model.X1 = model.A * model.B
    model.X2 = model.A | model.B
    model.X3 = model.A ^ model.B
    model.X4 = model.B - model.A
    model.X5 = model.B & model.A

model.Y = Set(initialize=model.B - model.A)
model.Y.add('foo')

if 1 > 0:
    instance = model.create()
    print("X1", len(instance.X1))
    print(instance.X1.data())
    print("X2", len(instance.X2))
    print(instance.X2.data())
    print("X3", len(instance.X3))
    print(instance.X3.data())
    print("X4", len(instance.X4))
    print(instance.X4.data())
    print("X5", len(instance.X5))
    print(instance.X5.data())

    print(instance.Y.data())
