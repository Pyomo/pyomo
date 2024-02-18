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

import pyomo.kernel as pmo

v = pmo.variable(value=2)

#
# Objectives
#

o = pmo.objective()
assert o() == None
assert o.expr == None

o = pmo.objective(expr=v**2 + 1)
assert o() == 5
assert pmo.value(o) == 5
assert pmo.value(o.expr) == 5

o = pmo.objective()
o.expr = v - 1
assert pmo.value(o) == 1

osub = pmo.objective(expr=v + 1)
o = pmo.objective(expr=osub + 1)
assert pmo.value(osub) == 3
assert pmo.value(o) == 4

osub.expr = v - 1
assert pmo.value(osub) == 1
assert pmo.value(o) == 2
