import pyomo.environ as pyo

model = pyo.AbstractModel()

# @decl1:
model.A = pyo.Set(ordered=pyo.Set.SortedOrder)
# @:decl1

instance = model.create_instance('set_options.dat')
instance.pprint()
