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

model.Z = pyo.Set(initialize=[('A1', 'B1', 1), ('A2', 'B2', 2), ('A3', 'B3', 3)])
# model.Z = pyo.Set(dimen=3)
model.D = pyo.Param(model.Z)

instance = model.create_instance('ABCD2.dat')

print('Z ' + str(sorted(list(instance.Z.data()))))
print('D')
for key in sorted(instance.D.keys()):
    print(pyo.name(instance.D, key) + " " + str(pyo.value(instance.D[key])))
