import pyomo.core.kernel as pk

#
# List containers
#

vl = pk.variable_list(
    pk.variable() for i in range(10))

cl = pk.constraint_list()
for i in range(10):
    cl.append(pk.constraint(vl[-1] == 1))

cl.insert(0, pk.constraint(vl[0]**2 >= 1))

del cl[0]

#
# Dict containers
#

# uses OrderedDict when ordered=True
vd = pk.variable_dict(
    ((str(i), pk.variable()) for i in range(10)),
    ordered=True)

cd = pk.constraint_dict(
    (i, pk.constraint(v == 1)) for i,v in vd.items())

cd = pk.constraint_dict()
for i, v in vd.items():
    cd[i] = pk.constraint(v == 1)

cd = pk.constraint_dict()
cd.update((i, pk.constraint()) for i,v in vd.items())

cd[None] = pk.constraint()

del cd[None]

#
# Nesting containers
#

b = pk.block()
b.bd = pk.block_dict()
b.bd[None] = pk.block_dict()
b.bd[None][1] = pk.block()
b.bd[None][1].x = pk.variable()
b.bd['a'] = pk.block_list()
b.bd['a'].append(pk.block())
