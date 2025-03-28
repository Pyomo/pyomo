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

model.A = pyo.Set()
model.Y = pyo.Param(model.A)

instance = model.create_instance('import2.tab.dat')

print('A ' + str(sorted(list(instance.A.data()))))
print('Y')
keys = instance.Y.keys()
for key in sorted(keys):
    print(str(key) + " " + str(pyo.value(instance.Y[key])))
