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

# written by jds, adapted for doc by dlw
import pyomo.environ as pyo

# simple way to get arbitrary, unique values for each thing
val_iter = 0


def get_val(*args, **kwds):
    global val_iter
    val_iter += 1
    return val_iter


model = pyo.ConcreteModel()
model.I = pyo.RangeSet(3)
model.x = pyo.Var(initialize=get_val)
model.y = pyo.Var(model.I, initialize=get_val)

model.b = pyo.Block()
model.b.a = pyo.Var(initialize=get_val)
model.b.b = pyo.Var(model.I, initialize=get_val)


def c_rule(b, i):
    b.c = pyo.Var(initialize=get_val)
    b.d = pyo.Var(b.model().I, initialize=get_val)


model.c = pyo.Block([1, 2], rule=c_rule)

model.pprint()

# @compprintloop
for v in model.component_objects(pyo.Var, descend_into=True):
    print("FOUND VAR:" + v.name)
    v.pprint()

for v_data in model.component_data_objects(pyo.Var, descend_into=True):
    print("Found: " + v_data.name + ", value = " + str(pyo.value(v_data)))
# @compprintloop
