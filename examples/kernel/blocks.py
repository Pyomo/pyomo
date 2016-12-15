import pyomo.core.kernel as pk

#
# Blocks
#

# define a simple optimization model
b = pk.block()
b.x = pk.variable()
b.c = pk.constraint(expr= b.x >= 1)
b.o = pk.objective(expr= b.x)

# define an optimization model with indexed containers
b = pk.block()

b.p = pk.parameter()
b.plist = pk.parameter_list(pk.parameter()
                            for i in range(10))
b.pdict = pk.parameter_dict(((i,j), pk.parameter())
                            for i in range(10)
                            for j in range(10))

b.x = pk.variable()
b.xlist = pk.variable_list(pk.variable()
                           for i in range(10))
b.xdict = pk.variable_dict(((i,j), pk.variable())
                           for i in range(10)
                           for j in range(10))

b.c = pk.constraint(b.x >= 1)
b.clist = pk.constraint_list(
    pk.constraint(b.xlist[i] >= i)
    for i in range(10))
b.cdict = pk.constraint_dict(
    ((i,j), pk.constraint(b.xdict[i,j] >= i * j))
    for i in range(10)
    for j in range(10))

b.o = pk.objective(
    b.x + sum(b.xlist) + sum(b.xdict.values()))

#
# Define a custom StaticBlock
#

class Widget(pk.StaticBlock):
    __slots__ = ("p", "input", "output", "c")
    def __init__(self, p, input=None):
        super(Widget, self).__init__()
        self.p = pk.parameter(value=p)
        self.input = pk.expression(expr=input)
        self.output = pk.variable()
        self.c = pk.constraint(
            self.output == self.input**2 / self.p)

b = pk.block()
b.x = pk.variable()
b.widgets = pk.block_list()
for i in range(10):
    b.widgets.append(Widget(i, input=b.x))
