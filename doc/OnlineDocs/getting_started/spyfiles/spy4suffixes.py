"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for suffixes.rst in testable form
"""

# @Declare_suffix_component
>>> from pyomo.environ import *
>>>
>>> model = ConcreteModel()
>>>
>>> # Export integer data
>>> model.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
>>>
>>> # Export and import floating point data
>>> model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
>>>
>>> # Store floating point data
>>> model.junk = Suffix()
# @Declare_suffix_component

# @Use_suffix_component_class_methods
>>> from pyomo.environ import *
>>>
>>> model = ConcreteModel()
>>> model.x = Var()
>>> model.y = Var([1,2,3])
>>> model.foo = Suffix()
# @Use_suffix_component_class_methods

# @Add_entries_to_suffix_declaration
>>> # Assign a suffix value of 1.0 to model.x
>>> model.foo.setValue(model.x, 1.0)
>>>
>>> # Same as above with dict interface
>>> model.foo[model.x] = 1.0
>>>
>>>
>>> # Assign a suffix value of 0.0 to all indices of model.y
>>> # By default this expands so that entries are created for
>>> # every index (y[1], y[2], y[3]) and not model.y itself
>>> model.foo.setValue(model.y, 0.0)
>>>
>>> # The same operation using the dict interface results in an entry only
>>> # for the parent component model.y
>>> model.foo[model.y] = 50.0
>>>
>>>
>>> # Assign a suffix value of -1.0 to model.y[1]
>>> model.foo.setValue(model.y[1], -1.0)
>>>
>>> # Same as above with the dict interface
>>> model.foo[model.y[1]] = -1.0
# @Add_entries_to_suffix_declaration



# @Print_value
>>> print(model.foo.get(model.x))         # -> 1.0
>>> print(model.foo[model.x])             # -> 1.0
>>>
>>> print(model.foo.get(model.y[1]))      # -> -1.0
>>> print(model.foo[model.y[1]])          # -> -1.0
>>>
>>> print(model.foo.get(model.y[2]))      # -> 0.0
>>> print(model.foo[model.y[2]])          # -> 0.0
>>>
>>> print(model.foo.get(model.y))         # -> 50.0
>>> print(model.foo[model.y])             # -> 50.0
>>>
>>> del model.foo[model.y]
>>>
>>> print(model.foo.get(model.y))         # -> None
>>> print(model.foo[model.y])             # -> raise KeyError
# @Print_value


# @Clear_value
>>> model.foo.clearValue(model.y)
>>>
>>> print(model.foo[model.y[1]])          # -> raise KeyError
>>>
>>> del model.foo[model.y[1]]             # -> raise KeyError
>>>
>>> model.foo.clearValue(model.y[1])      # -> does nothing
# @Clear_value

# @Import_suffix_information
>>> from pyomo.environ import *
>>>
>>> model = ConcreteModel()
>>> model.dual = Suffix(direction=Suffix.IMPORT)
>>> model.x = Var()
>>> model.obj = Objective(expr=model.x)
>>> model.con = Constraint(expr=model.x>=1.0)
# @Import_suffix_information

# @Print_dual_value
>>> print(instance.dual[instance.con]) # -> 1.0
# @Print_dual_value

# @Export_suffix_data
>>> from pyomo.environ import *
>>>
>>> model = ConcreteModel()
>>> model.y = Var([1,2,3],within=NonNegativeReals)
>>>
>>> model.sosno = Suffix(direction=Suffix.EXPORT)
>>> model.ref = Suffix(direction=Suffix.EXPORT)
>>>
>>> # Add entry for each index of model.y
>>> model.sosno.setValue(model.y,1)
>>> model.ref[model.y[1]] = 0
>>> model.ref[model.y[2]] = 1
>>> model.ref[model.y[3]] = 2
# @Export_suffix_data

# @Suffix_initialization_rule_keyword
>>> from pyomo.environ import *
>>>
>>> model = AbstractModel()
>>> model.x = Var()
>>> model.c = Constraint(expr= model.x >= 1)
>>>
>>> def foo_rule(m):
>>>    return ((m.x, 2.0), (m.c, 3.0))
>>> model.foo = Suffix(rule=foo_rule)
>>>
>>> # Instantiate the model
>>> inst = model.create()
>>> print(inst.foo[model.x]) # -> raise KeyError
>>> print(inst.foo[inst.x])  # -> 2.0
>>> print(inst.foo[inst.c])  # -> 3.0
# @Suffix_initialization_rule_keyword
