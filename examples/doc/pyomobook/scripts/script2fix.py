from pyomo.opt import SolverFactory
from multimodal import model as instance

# @value:
instance.y = 3.5
instance.x.value = 3.5
# @:value
# @fixed:
instance.y.fixed = True
instance.y.fix()
instance.y.fix(3.5)
# @:fixed

solver.solve(instance)

print("First   x was %f and y was %f"
      % (instance.x.value, instance.y.value))

instance.x.fixed = True
instance.y.fixed = False

solver.solve(instance)

print("Next    x was %f and y was %f"
      % (instance.x.value, instance.y.value))

instance.x.fixed = False
instance.y.fixed = True

solver.solve(instance)

print("Finally x was %f and y was %f"
      % (instance.x.value, instance.y.value))
