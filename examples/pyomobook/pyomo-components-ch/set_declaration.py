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

model = pyo.AbstractModel()

# @decl1:
model.A = pyo.Set()
# @:decl1

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = pyo.AbstractModel()

# @decl2:
model.A = pyo.Set()
model.B = pyo.Set()
model.C = pyo.Set(model.A)
model.D = pyo.Set(model.A, model.B)
# @:decl2

model = pyo.AbstractModel()
instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = pyo.AbstractModel()

# @decl6:
model.E = pyo.Set([1, 2, 3])
f = set([1, 2, 3])
model.F = pyo.Set(f)
# @:decl6

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = pyo.AbstractModel()
model = pyo.AbstractModel()

# @decl3:
model.A = pyo.Set()
model.B = pyo.Set()
model.G = model.A | model.B  # set union
model.H = model.B & model.A  # set intersection
model.I = model.A - model.B  # set difference
model.J = model.A ^ model.B  # set exclusive-or
# @:decl3

instance = model.create_instance('set_declaration.dat')
instance.pprint()
model = pyo.AbstractModel()

# @decl4:
model.A = pyo.Set()
model.B = pyo.Set()
model.K = model.A * model.B
# @:decl4
# Removing this component, which we're going to read again
model.del_component('K')

instance = model.create_instance('set_declaration.dat')
instance.pprint(verbose=True)
