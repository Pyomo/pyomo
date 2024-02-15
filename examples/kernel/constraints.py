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

v = pmo.variable()

#
# Equality constraints
#

c = pmo.constraint(v == 1)

c = pmo.constraint(expr=v == 1)

c = pmo.constraint(body=v, rhs=1)

c = pmo.constraint()
c.body = v
c.rhs = 1

#
# Single-sided inequality constraints
#

c = pmo.constraint(v <= 1)

c = pmo.constraint(expr=v <= 1)

c = pmo.constraint(body=v, ub=1)

c = pmo.constraint()
c.body = v
c.ub = 1

c = pmo.constraint(v >= 1)

c = pmo.constraint(expr=v >= 1)

c = pmo.constraint(body=v, lb=1)

c = pmo.constraint()
c.body = v
c.lb = 1

#
# Range constraints
#

c = pmo.constraint((0, v, 1))

c = pmo.constraint(expr=(0, v, 1))

c = pmo.constraint(lb=0, body=v, ub=1)

c = pmo.constraint()
c.lb = 0
c.body = v
c.ub = 1

#
# Usage
#

v.value = 2

# initialize a range constraint
r = pmo.constraint(lb=0, body=v**2, ub=5)
assert pmo.value(r.lb) == 0
assert pmo.value(r.ub) == 5
assert pmo.value(r.body) == 4
assert r.equality == False
assert r.lslack == 4
assert r.uslack == 1
assert r.slack == 1

# change the lb
r.lb = -1
assert r.lslack == 5
assert r.uslack == 1
assert r.slack == 1

# setting rhs turns this into an equality constraint
r.rhs = 2
assert pmo.value(r.lb) == 2
assert pmo.value(r.ub) == 2
assert pmo.value(r.rhs) == 2
assert pmo.value(r.body) == 4
assert r.equality == True
assert r.lslack == 2
assert r.uslack == -2
assert r.slack == -2
