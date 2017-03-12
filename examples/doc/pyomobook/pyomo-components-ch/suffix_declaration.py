from pyomo.environ import *

print('')
print('*** suffixsimple ***')
model = ConcreteModel()
# @suffixsimple:
model.foo = Suffix()
# @:suffixsimple

model = ConcreteModel()

# @suffixdecl:
# Export integer data
model.priority = Suffix(direction=Suffix.EXPORT, 
                        datatype=Suffix.INT)

# Export and import floating point data
model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

# @:suffixdecl
model.pprint()

# @suffixinitrule:
model = AbstractModel()
model.x = Var()
model.c = Constraint(expr=model.x >= 1)

def foo_rule(m):
   return ((m.x, 2.0), (m.c, 3.0))
model.foo = Suffix(initialize=foo_rule)
# @:suffixinitrule
model.pprint()

del foo_rule # Needed to avoid implicit rule warning in next example

print('')
print('*** suffix1 ***')
# @suffix1:
model = ConcreteModel()
model.x = Var()
model.y = Var([1,2,3], dense=True)
model.foo = Suffix()
# @:suffix1
print('suffix1a')
# @suffix1a:
# Assign the value 1.0 to suffix 'foo' for model.x
model.x.set_suffix_value('foo', 1.0)

# Assign the value 2.0 to suffix model.foo for model.x
model.x.set_suffix_value(model.foo, 2.0)

# Get the value of suffix 'foo' for model.x
print(model.x.get_suffix_value('foo'))          # 2.0
# @:suffix1a
print('suffix1b')
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
print('suffix1c')
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
