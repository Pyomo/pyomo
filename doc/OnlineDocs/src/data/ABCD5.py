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

model = AbstractModel()

# @decl
model.Z = Set()
model.Y = Param(model.Z)
model.W = Param(model.Z)
# @decl

instance = model.create_instance('ABCD5.dat')

print('Z ' + str(sorted(list(instance.Z.data()))))
print('Y')
for key in sorted(instance.Y.keys()):
    print(name(instance.Y, key) + " " + str(value(instance.Y[key])))
print('W')
for key in sorted(instance.W.keys()):
    print(name(instance.W, key) + " " + str(value(instance.W[key])))
