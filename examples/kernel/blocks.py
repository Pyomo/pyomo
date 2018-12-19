import pyomo.kernel as pmo

#
# Blocks
#

# define a simple optimization model
b = pmo.block()
b.x = pmo.variable()
b.c = pmo.constraint(expr= b.x >= 1)
b.o = pmo.objective(expr= b.x)

# define an optimization model with indexed containers
b = pmo.block()

b.p = pmo.parameter()
b.plist = pmo.parameter_list(pmo.parameter()
                             for i in range(10))
b.pdict = pmo.parameter_dict(((i,j), pmo.parameter())
                             for i in range(10)
                             for j in range(10))

b.x = pmo.variable()
b.xlist = pmo.variable_list(pmo.variable()
                            for i in range(10))
b.xdict = pmo.variable_dict(((i,j), pmo.variable())
                            for i in range(10)
                            for j in range(10))

b.c = pmo.constraint(b.x >= 1)
b.clist = pmo.constraint_list(
    pmo.constraint(b.xlist[i] >= i)
    for i in range(10))
b.cdict = pmo.constraint_dict(
    ((i,j), pmo.constraint(b.xdict[i,j] >= i * j))
    for i in range(10)
    for j in range(10))

b.o = pmo.objective(
    b.x + sum(b.xlist) + sum(b.xdict.values()))

#
# Define a custom block
#

class Widget(pmo.block):
    def __init__(self, p, input=None):
        super(Widget, self).__init__()
        self.p = pmo.parameter(value=p)
        self.input = pmo.expression(expr=input)
        self.output = pmo.variable()
        self.c = pmo.constraint(
            self.output == self.input**2 / self.p)

b = pmo.block()
b.x = pmo.variable()
b.widgets = pmo.block_list()
for i in range(10):
    b.widgets.append(Widget(i, input=b.x))
