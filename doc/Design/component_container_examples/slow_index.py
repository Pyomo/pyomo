import time
from pyomo.environ import *

m = ConcreteModel()
m.Edges = [(i,j) for i in range(500) for j in range(500)]
m.Commodities = [(1,2), (3,2)]
m.Flow = Var(m.Commodities, m.Edges)

# make sure the variable is constructed
[m.Flow[s,t,u,v]
 for (s,t) in m.Commodities
 for (u,v) in m.Edges]
assert len(m.Flow) > 0

start = time.time()
for c in m.Commodities:
    for e in m.Edges:
        x = m.Flow[c,e].value
print(time.time() - start)

start = time.time()
for c in m.Commodities:
    for e in m.Edges:
        x = m.Flow[c[0],c[1],e[0],e[1]].value
print(time.time() - start)
