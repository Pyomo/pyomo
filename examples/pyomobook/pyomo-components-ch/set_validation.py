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

model.A = pyo.Set()
# @decl1:
model.B = pyo.Set(within=model.A)
# @:decl1


# @decl2:
def C_validate(model, value):
    return value in model.A


model.C = pyo.Set(validate=C_validate)
# @:decl2

instance = model.create_instance('set_validation.dat')
instance.pprint(verbose=True)
