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

# @mutable1:
model = pyo.ConcreteModel()
p = {1: 1, 2: 4, 3: 9}

model.A = pyo.Set(initialize=[1, 2, 3])
model.p = pyo.Param(model.A, initialize=p)
model.x = pyo.Var(model.A, within=pyo.NonNegativeReals)

model.o = pyo.Objective(expr=sum(model.p[i] * model.x[i] for i in model.A))

# @:mutable1
model.pprint()

# @mutable2:
model = pyo.ConcreteModel()
p = {1: 1, 2: 4, 3: 9}

model.A = pyo.Set(initialize=[1, 2, 3])
model.p = pyo.Param(model.A, initialize=p, mutable=True)
model.x = pyo.Var(model.A, within=pyo.NonNegativeReals)

model.o = pyo.Objective(expr=pyo.summation(model.p, model.x))

model.p[2] = 4.2
model.p[3] = 3.14
# @:mutable2
model.pprint()
