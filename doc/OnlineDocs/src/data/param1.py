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
model.A = Param()
model.B = Param()
model.C = Param()
model.D = Param()
model.E = Param()
# @decl

instance = model.create_instance('param1.dat')

print(value(instance.A))
print(value(instance.B))
print(value(instance.C))
print(value(instance.D))
print(value(instance.E))
