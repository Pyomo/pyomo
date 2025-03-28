#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

model = pyo.AbstractModel()
model.I = pyo.RangeSet(1, 4)
model.x = pyo.Var(model.I)


def c_rule(m, i):
    return m.x[i] >= i


model.c = pyo.Constraint(model.I, rule=c_rule)


def foo_rule(m):
    return ((m.x[i], 3.0 * i) for i in m.I)


model.foo = pyo.Suffix(rule=foo_rule)

# instantiate the model
inst = model.create_instance()
for i in inst.I:
    print(i, inst.foo[inst.x[i]])
