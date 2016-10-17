from pyomo.environ import *

model = ConcreteModel()
model.s = Set(initialize=[1,2,3])
model.x = Var()
model.y = Var(model.s)

# simulate a solve
model.x.value = 10
model.y[1].value = 1
model.y[2].value = 2
model.y[3].value = 3

# @loop1:
from pyomo.core import Var
print("\nLoop By Variable Objects")
for var in model.component_data_objects(Var):
    print("%s: %f" % (var, var()))
# @:loop1

# @loop2:
print("\nLoop Over Indexed Variable:")
for idx in model.y:
    print("%s %s" % (model.y[idx], model.y[idx]()))
# @:loop2

# @loop3:
from pyomo.core import Var
print("\nLoop by Variable Containers")
for var_container in model.component_objects(Var):
    print(var_container)
    for index in var_container:
        print("  %s: %f" % (index, var_container[index]()))
# @:loop3
