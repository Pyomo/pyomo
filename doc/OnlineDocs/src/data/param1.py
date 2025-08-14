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
model.A = pyo.Param()
model.B = pyo.Param()
model.C = pyo.Param()
model.D = pyo.Param()
model.E = pyo.Param()
# @decl

instance = model.create_instance('param1.dat')

print(pyo.value(instance.A))
print(pyo.value(instance.B))
print(pyo.value(instance.C))
print(pyo.value(instance.D))
print(pyo.value(instance.E))
