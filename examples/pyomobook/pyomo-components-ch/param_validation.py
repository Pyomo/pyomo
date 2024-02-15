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

import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl1:
model.Z = pyo.Param(within=pyo.Reals)
# @:decl1


# @decl2:
def Y_validate(model, value):
    return value in pyo.Reals


model.Y = pyo.Param(validate=Y_validate)
# @:decl2

# @decl3:
model.A = pyo.Set(initialize=[1, 2, 3])


def X_validate(model, value, i):
    return value > i


model.X = pyo.Param(model.A, validate=X_validate)
# @:decl3

instance = model.create_instance('param_validation.dat')
instance.pprint()
