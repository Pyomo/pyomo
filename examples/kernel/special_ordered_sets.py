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

v1 = pmo.variable()
v2 = pmo.variable()
v3 = pmo.variable()

#
# Special Ordered Sets (Type 1)
#

s = pmo.sos([v1, v2])
assert s.level == 1
assert s.weights == (1, 2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pmo.sos([v1, v2], level=1)
assert s.level == 1
assert s.weights == (1, 2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pmo.sos1([v1, v2])
assert s.level == 1
assert s.weights == (1, 2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

#
# Special Ordered Sets (Type 2)
#

s = pmo.sos([v1, v2], level=2)
assert s.level == 2
assert s.weights == (1, 2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

s = pmo.sos2([v1, v2])
assert s.level == 2
assert s.weights == (1, 2)
assert len(s.variables) == 2
assert v1 in s
assert v2 in s

#
# Special Ordered Sets (Type n)
#

s = pmo.sos([v1, v2, v3], level=3)
assert s.level == 3
assert s.weights == (1, 2, 3)
assert len(s.variables) == 3
assert v1 in s
assert v2 in s
assert v3 in s

#
# Specifying weights
#

# using known values
s = pmo.sos([v1, v2], weights=[1.2, 2.5])
assert s.weights == (1.2, 2.5)

# using parameters
p = pmo.parameter_list(pmo.parameter() for i in range(2))
s = pmo.sos([v1, v2], weights=[p[0] ** 2, p[1] ** 2])
assert len(s.weights) == 2
p[0].value = 1
p[1].value = 2
assert tuple(pmo.value(w) for w in s.weights) == (1, 4)

# using data expressions
d = pmo.expression_list(pmo.data_expression() for i in range(2))
s = pmo.sos([v1, v2], weights=d)
assert len(s.weights) == 2
d[0].expr = p[0] + 1
d[1].expr = p[0] + p[1]
assert tuple(pmo.value(w) for w in s.weights) == (2, 3)

#
# Example model (discontiguous variable domain)
#

domain = [-1.1, 4.49, 8.1, -30.2, 12.5]

m = pmo.block()

m.z = pmo.variable_list(pmo.variable(lb=0) for i in range(len(domain)))
m.y = pmo.variable()

m.o = pmo.objective(m.y, sense=pmo.maximize)

m.c1 = pmo.constraint(m.y == sum(v * z for v, z in zip(m.z, domain)))
m.c2 = pmo.constraint(sum(m.z) == 1)
m.s = pmo.sos1(m.z)
