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
# Continuous variables
#

v = pmo.variable()

v = pmo.variable(domain=pmo.Reals)

v = pmo.variable(domain=pmo.NonNegativeReals, ub=10)

v = pmo.variable(domain_type=pmo.RealSet, lb=1)

# error (because domain lower bound is finite)
# v = pmo.variable(domain=pmo.NonNegativeReals,
#                 lb=1)

#
# Discrete variables
#

v = pmo.variable(domain=pmo.Binary)

v = pmo.variable(domain=pmo.Integers)

v = pmo.variable(domain=pmo.NonNegativeIntegers, ub=10)

v = pmo.variable(domain_type=pmo.IntegerSet, lb=1)

# error (because domain upper bound is finite)
# v = pmo.variable(domain=pmo.NegativeIntegers,
#                 ub=10)

#
# Usage
#

v = pmo.variable()
assert v.value == None
assert v.lb == None
assert v.ub == None
assert v.fixed == False
assert v.domain_type == pmo.RealSet

# set the value
v.value = 2
assert v.value == 2
assert pmo.value(v**2) == 4

# set the bounds
v.lb = 10
v.ub = 20
assert v.lb == 10
assert v.ub == 20

# set the domain (always overwrites bounds, even if infinite)
v.domain = pmo.Reals
assert v.lb == None
assert v.ub == None
assert v.domain_type == pmo.RealSet
v.domain = pmo.Binary
assert v.lb == 0
assert v.ub == 1
assert v.domain_type == pmo.IntegerSet

# set the domain_type (never overwrites bounds)
v.domain_type = pmo.RealSet
assert v.lb == 0
assert v.ub == 1
assert v.domain_type == pmo.RealSet

# fix the variable to its current value
v.fix()
assert v.value == 2
assert v.fixed == True

# fix the variable to a new value
v.fix(1)
assert v.value == 1
assert v.fixed == True

# free the variable
v.free()
assert v.value == 1
assert v.fixed == False
