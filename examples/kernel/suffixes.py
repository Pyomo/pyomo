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

import pyomo.kernel as pmo

#
# Suffixes
#

# collect dual information when the model is solved
b = pmo.block()
b.x = pmo.variable()
b.c = pmo.constraint(expr=b.x >= 1)
b.o = pmo.objective(expr=b.x)
b.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

# suffixes behave as dictionaries that map
# components to values
s = pmo.suffix()
assert len(s) == 0

v = pmo.variable()
s[v] = 2
assert len(s) == 1
assert bool(v in s) == True
assert s[v] == 2

# error (a dict / list container is not a component)
vlist = pmo.variable_list()
s[vlist] = 1
