from pyomo.environ import value
from pyomo.opt import SolverFactory
from concrete1 import model as instance1
from concrete2 import model as instance2

with SolverFactory("glpk") as opt:
    results1 = opt.solve(instance1)
    results2 = opt.solve(instance2)

print("x_2  value: %s" % (instance1.x_2.value))
print("x_2  value: %s" % (instance1.x_2()))
print("x_2  value: %s" % (value(instance1.x_2)))
print("x[2] value: %s" % (instance2.x[2].value))
print("x[2] value: %s" % (instance2.x[2]()))
print("x[2] value: %s" % (value(instance2.x[2])))

print("x_2  object: %s" % (instance1.x_2))
print("x[2] object: %s" % (instance2.x[2]))

if value(instance1.x_2) == value(instance2.x[2]):
    print("x_2 == x[2]")
else:
    print("x_2 != x[2]")

if instance1.x_2 == instance2.x[2]:
    print("x_2 == x[2]")
else:
    print("x_2 != x[2]")
