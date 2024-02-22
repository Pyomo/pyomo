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

# munson1.py
import pyomo.environ as pyo
from pyomo.mpec import Complementarity, complements


model = pyo.ConcreteModel()

model.x1 = pyo.Var()
model.x2 = pyo.Var()
model.x3 = pyo.Var()

model.f1 = Complementarity(
    expr=complements(model.x1 >= 0, model.x1 + 2 * model.x2 + 3 * model.x3 >= 1)
)

model.f2 = Complementarity(expr=complements(model.x2 >= 0, model.x2 - model.x3 >= -1))

model.f3 = Complementarity(expr=complements(model.x3 >= 0, model.x1 + model.x2 >= -1))
