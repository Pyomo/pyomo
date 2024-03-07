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
model.A = Set()
model.B = Set(model.A)
# @decl
# model.C = Set(model.A,model.A)

instance = model.create_instance('set3.dat')

print(sorted(list(instance.A.data()), key=lambda x: x if type(x) is str else str(x)))
print(sorted(list(instance.B[1].data()), key=lambda x: x if type(x) is str else str(x)))
print(
    sorted(
        list(instance.B['aaa'].data()), key=lambda x: x if type(x) is str else str(x)
    )
)
