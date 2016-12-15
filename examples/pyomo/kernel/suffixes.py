import pyomo.core.kernel as pk

#
# Suffixes
#

# collect dual information when the model is solved
b = pk.block()
b.x = pk.variable()
b.c = pk.constraint(expr= b.x >= 1)
b.o = pk.objective(expr= b.x)
b.dual = pk.suffix(direction=pk.suffix.IMPORT)

# suffixes behave as dictionaries that map
# components to values
s = pk.suffix()
assert len(s) == 0

v = pk.variable()
s[v] = 2
assert len(s) == 1
assert bool(v in s) == True
assert s[v] == 2

# error (a dict / list container is not a component)
vlist = pk.variable_list()
s[vlist] = 1
