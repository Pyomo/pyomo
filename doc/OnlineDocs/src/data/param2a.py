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
model.A = pyo.Set()
model.B = pyo.Param(model.A)
# @decl

instance = model.create_instance('param2a.dat')

keys = instance.B.keys()
for key in sorted(keys):
    print(str(key) + " " + str(pyo.value(instance.B[key])))
