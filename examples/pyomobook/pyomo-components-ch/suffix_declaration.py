#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo

print('')
print('*** suffixsimple ***')
model = pyo.ConcreteModel()
# @suffixsimple:
model.foo = pyo.Suffix()
# @:suffixsimple

model = pyo.ConcreteModel()

# @suffixdecl:
# Export integer data
model.priority = pyo.Suffix(direction=pyo.Suffix.EXPORT, datatype=pyo.Suffix.INT)

# Export and import floating point data
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

# @:suffixdecl
model.pprint()

# @suffixinitrule:
model = pyo.AbstractModel()
model.x = pyo.Var()
model.c = pyo.Constraint(expr=model.x >= 1)


def foo_rule(m):
    return ((m.x, 2.0), (m.c, 3.0))


model.foo = pyo.Suffix(initialize=foo_rule)
# @:suffixinitrule
model.pprint()

del foo_rule  # Needed to avoid implicit rule warning in next example

print('')
print('*** suffix1 ***')
# @suffix1:
model = pyo.ConcreteModel()
model.x = pyo.Var()
model.y = pyo.Var([1, 2, 3])
model.foo = pyo.Suffix()
# @:suffix1
print('suffix1a')
# @suffix1a:
# Assign the value 1.0 to suffix 'foo' for model.x
model.x.set_suffix_value('foo', 1.0)

# Assign the value 2.0 to suffix model.foo for model.x
model.x.set_suffix_value(model.foo, 2.0)

# Get the value of suffix 'foo' for model.x
print(model.x.get_suffix_value('foo'))  # 2.0
# @:suffix1a
print('suffix1b')
# @suffix1b:
# Assign the value 3.0 to suffix model.foo for model.y
model.y.set_suffix_value(model.foo, 3.0)

# Assign the value 4.0 to suffix model.foo for model.y[2]
model.y[2].set_suffix_value(model.foo, 4.0)

# Get the value of suffix 'foo' for model.y
print(model.y.get_suffix_value(model.foo))  # None
print(model.y[1].get_suffix_value(model.foo))  # 3.0
print(model.y[2].get_suffix_value(model.foo))  # 4.0
print(model.y[3].get_suffix_value(model.foo))  # 3.0
# @:suffix1b

# @suffix1d:
model.y[3].clear_suffix_value(model.foo)

print(model.y.get_suffix_value(model.foo))  # None
print(model.y[1].get_suffix_value(model.foo))  # 3.0
print(model.y[2].get_suffix_value(model.foo))  # 4.0
print(model.y[3].get_suffix_value(model.foo))  # None
# @:suffix1d
