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
# List containers
#

vl = pmo.variable_list(pmo.variable() for i in range(10))

cl = pmo.constraint_list()
for i in range(10):
    cl.append(pmo.constraint(vl[-1] == 1))

cl.insert(0, pmo.constraint(vl[0] ** 2 >= 1))

del cl[0]

#
# Dict containers
#

vd = pmo.variable_dict(((str(i), pmo.variable()) for i in range(10)))

cd = pmo.constraint_dict((i, pmo.constraint(v == 1)) for i, v in vd.items())

cd = pmo.constraint_dict()
for i, v in vd.items():
    cd[i] = pmo.constraint(v == 1)

cd = pmo.constraint_dict()
cd.update((i, pmo.constraint()) for i, v in vd.items())

cd[None] = pmo.constraint()

del cd[None]

#
# Nesting containers
#

b = pmo.block()
b.bd = pmo.block_dict()
b.bd[None] = pmo.block_dict()
b.bd[None][1] = pmo.block()
b.bd[None][1].x = pmo.variable()
b.bd['a'] = pmo.block_list()
b.bd['a'].append(pmo.block())
