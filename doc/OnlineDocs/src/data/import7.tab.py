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

model.I = Set(initialize=['I1', 'I2', 'I3', 'I4'])
model.A = Set(initialize=['A1', 'A2', 'A3'])
model.U = Param(model.I, model.A)
# BUG:  This should cause an error
# model.U = Param(model.A,model.I)

instance = model.create_instance('import7.tab.dat')

print('I ' + str(sorted(list(instance.I.data()))))
print('A ' + str(sorted(list(instance.A.data()))))
print('U')
for key in sorted(instance.U.keys()):
    print(name(instance.U, key) + " " + str(value(instance.U[key])))
