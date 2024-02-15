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
model.A = pyo.RangeSet(10)
# @:decl1

# @decl3:
model.C = pyo.RangeSet(5, 10)
# @:decl3

# @decl4:
model.D = pyo.RangeSet(2.5, 11, 1.5)
# @:decl4

instance = model.create_instance()
instance.pprint(verbose=True)
