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

model.I = pyo.Set(initialize=['I1', 'I2', 'I3', 'I4'])
model.A = pyo.Set(initialize=['A1', 'A2', 'A3'])
model.U = pyo.Param(model.A, model.I)

instance = model.create_instance('import8.tab.dat')

print('A ' + str(sorted(list(instance.A.data()))))
print('I ' + str(sorted(list(instance.I.data()))))
print('U')
for key in sorted(instance.U.keys()):
    print(pyo.name(instance.U, key) + " " + str(pyo.value(instance.U[key])))
