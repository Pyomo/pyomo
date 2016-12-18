import pyomo.core.kernel as pk

#
# Piecewise linear constraints
#

breakpoints = [1,2,3,4]
values = [1,2,1,2]

x = pk.variable()
y = pk.variable()
p = pk.piecewise(breakpoints,
                 values,
                 input=x,
                 output=y,
                 repn='sos2',
                 bound='eq')


# change the input and output variables
z = pk.variable()
q = pk.variable()
p.set_input(z)
p.set_output(q)

# evalute the function
assert p(1) == 1
assert p(1.5) == 1.5
assert p(2) == 2
assert p(2.5) == 1.5
assert p(3) == 1
assert p(2.5) == 1.5
assert p(4) == 2
