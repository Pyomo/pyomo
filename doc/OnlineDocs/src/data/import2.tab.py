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

model.A = Set()
model.Y = Param(model.A)

instance = model.create_instance('import2.tab.dat')

print('A ' + str(sorted(list(instance.A.data()))))
print('Y')
keys = instance.Y.keys()
for key in sorted(keys):
    print(str(key) + " " + str(value(instance.Y[key])))
