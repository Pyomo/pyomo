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

#
# Mutable parameters
#

p = pmo.parameter()
assert p.value == None

p = pmo.parameter(value=2)
assert p.value == 2

p.value = 4
assert p.value == 4
assert pmo.value(p**2) == 16
assert pmo.value(p - 1) == 3

v = pmo.variable()
c = pmo.constraint((p - 1, v, p + 1))
assert pmo.value(c.lb) == 3
assert pmo.value(c.ub) == 5

p.value = -1
assert pmo.value(c.lb) == -2
assert pmo.value(c.ub) == 0
