from pyomo.environ import *

print('')
print('*** suffix1 ***')
# @suffix1:
model = ConcreteModel()
model.x = Var()
model.y = Var([1,2,3], dense=True)
model.foo = Suffix()
# @:suffix1
# @suffix1a:
# Assign the value 1.0 to suffix 'foo' for model.x
model.x.set_suffix_value('foo', 1.0)

# Assign the value 2.0 to suffix model.foo for model.x
model.x.set_suffix_value(model.foo, 2.0)

# Get the value of suffix 'foo' for model.x
print(model.x.get_suffix_value('foo'))          # 2.0
# @:suffix1a
# @suffix1b:
# Assign the value 3.0 to suffix model.foo for model.y
model.y.set_suffix_value(model.foo, 3.0)

# Assign the value 4.0 to suffix model.foo for model.y[2]
model.y[2].set_suffix_value(model.foo, 4.0)

# Get the value of suffix 'foo' for model.y
print(model.y.get_suffix_value(model.foo))      # None
print(model.y[1].get_suffix_value(model.foo))   # 3.0
print(model.y[2].get_suffix_value(model.foo))   # 4.0
print(model.y[3].get_suffix_value(model.foo))   # 3.0
# @:suffix1b
# @suffix1c:
# Assign the value 5.0 to suffix model.foo for just the
# model.y component
model.y.set_suffix_value(model.foo, 5.0, expand=False)

# Get the value of suffix 'foo' for model.y
print(model.y.get_suffix_value(model.foo))      # 5.0
print(model.y[1].get_suffix_value(model.foo))   # 3.0
print(model.y[2].get_suffix_value(model.foo))   # 4.0
print(model.y[3].get_suffix_value(model.foo))   # 3.0
# @:suffix1c
model.pprint()

# @suffix1d:
model.y.clear_suffix_value(model.foo, expand=False)
model.y[3].clear_suffix_value(model.foo)

print(model.y.get_suffix_value(model.foo))      # None
print(model.y[1].get_suffix_value(model.foo))   # 3.0
print(model.y[2].get_suffix_value(model.foo))   # 4.0
print(model.y[3].get_suffix_value(model.foo))   # None
# @:suffix1d


# @suffix2:
model = ConcreteModel()

# Export integer data
model.priority = Suffix(direction=Suffix.EXPORT, 
                        datatype=Suffix.INT)

# Export and import floating point data
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

# Store floating point data
model.junk = Suffix()
# @:suffix2
model.pprint()


# @suffix3:
model = AbstractModel()
model.x = Var()
model.c = Constraint(expr=model.x >= 1)

def foo_rule(m):
   return ((m.x, 2.0), (m.c, 3.0))
model.foo = Suffix(initialize=foo_rule)

inst = model.create_instance()
print(inst.x.get_suffix_value(inst.foo))    # 2
print(inst.c.get_suffix_value(inst.foo))    # 3
# @:suffix3
model.pprint()
del foo_rule

# @suffix4:
model = ConcreteModel()
model.x = Var()
model.y = Var([1,2,3], dense=True)
model.foo = Suffix()

# Assign a suffix value of 1.0 to model.x
model.foo.set_value(model.x, 1.0)

# Same as above with dict interface
model.foo[model.x] = 1.0

# Assign a suffix value of 0.0 to all indices of model.y
model.foo.set_value(model.y, 0.0)

# The same operation using the dict interface results
# in a suffix for the parent component model.y
model.foo[model.y] = 50.0


# Assign a suffix value of -1.0 to model.y[1]
model.foo.set_value(model.y[1], -1.0)

# Same as above with the dict interface
model.foo[model.y[1]] = -1.0
# @:suffix4
# @suffix5:
print(model.foo.get(model.x))         # -> 1.0
print(model.foo[model.x])             # -> 1.0

print(model.foo.get(model.y[1]))      # -> -1.0
print(model.foo[model.y[1]])          # -> -1.0

print(model.foo.get(model.y[2]))      # -> 0.0
print(model.foo[model.y[2]])          # -> 0.0

print(model.foo.get(model.y))         # -> 50.0
print(model.foo[model.y])             # -> 50.0

del model.foo[model.y]

print(model.foo.get(model.y))         # -> None
# @:suffix5

model.pprint()

