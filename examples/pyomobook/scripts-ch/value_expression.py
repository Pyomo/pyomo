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

model = pyo.ConcreteModel()
model.u = pyo.Var(initialize=2.0)

# unexpected expression instead of value
a = model.u - 1
print(a)  # "u - 1"
print(type(a))  # <class 'pyomo.core.expr.numeric_expr.LinearExpression'>

# correct way to access the value
b = pyo.value(model.u) - 1
print(b)  # 1.0
print(type(b))  # <class 'float'>
