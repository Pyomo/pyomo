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
# Piecewise linear constraints
#

breakpoints = [1, 2, 3, 4]
values = [1, 2, 1, 2]

x = pmo.variable(lb=1, ub=4)
y = pmo.variable()
p = pmo.piecewise(breakpoints, values, input=x, output=y, repn='sos2', bound='eq')

# change the input and output variables
z = pmo.variable(lb=1, ub=4)
q = pmo.variable()
p.input.expr = z
p.output.expr = q

# re-validate the function after changing inputs
# (will raise PiecewiseValidationError when validation fails)
p.validate()

# evaluate the function
assert p(1) == 1
assert p(1.5) == 1.5
assert p(2) == 2
assert p(2.5) == 1.5
assert p(3) == 1
assert p(2.5) == 1.5
assert p(4) == 2

breakpoints = [
    pmo.parameter(1),
    pmo.parameter(2),
    pmo.parameter(3),
    pmo.parameter(None),
]
values = [pmo.parameter(1), pmo.parameter(2), pmo.parameter(1), pmo.parameter(None)]
p = pmo.piecewise(
    breakpoints, values, input=x, output=y, repn='sos2', bound='eq', validate=False
)

# change the function parameters and
# validate that the inputs are correct
breakpoints[3].value = 4
values[3].value = 2
p.validate()

# evaluate the function
assert p(1) == 1
assert p(1.5) == 1.5
assert p(2) == 2
assert p(2.5) == 1.5
assert p(3) == 1
assert p(2.5) == 1.5
assert p(4) == 2

#
# Example model (piecewise linear objective)
#
breakpoints = [-1.0, 0.0, 1.0, 2.0]
function_points = [2.0, -2.5, 3.0, 1.0]

m = pmo.block()

m.x = pmo.variable(lb=-1, ub=2.0)
m.y = pmo.variable()

m.o = pmo.objective(m.y)

m.pw = pmo.piecewise(breakpoints, function_points, input=m.x, output=m.y, repn='inc')
