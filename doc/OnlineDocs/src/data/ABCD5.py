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

# @decl
model.Z = pyo.Set()
model.Y = pyo.Param(model.Z)
model.W = pyo.Param(model.Z)
# @decl

instance = model.create_instance('ABCD5.dat')

print('Z ' + str(sorted(list(instance.Z.data()))))
print('Y')
for key in sorted(instance.Y.keys()):
    print(pyo.name(instance.Y, key) + " " + str(pyo.value(instance.Y[key])))
print('W')
for key in sorted(instance.W.keys()):
    print(pyo.name(instance.W, key) + " " + str(pyo.value(instance.W[key])))
