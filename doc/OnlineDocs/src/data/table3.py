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
model.B = pyo.Set(initialize=['B1', 'B2', 'B3'])
model.Z = pyo.Set(dimen=2)

model.M = pyo.Param(model.A)
model.N = pyo.Param(model.A, model.B)


instance = model.create_instance('table3.dat')
instance.pprint()
