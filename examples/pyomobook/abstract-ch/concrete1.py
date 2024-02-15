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
model.x_1 = pyo.Var(within=pyo.NonNegativeIntegers)
model.x_2 = pyo.Var(within=pyo.NonNegativeIntegers)
model.obj = pyo.Objective(expr=model.x_1 + 2 * model.x_2)
model.con1 = pyo.Constraint(expr=3 * model.x_1 + 4 * model.x_2 >= 1)
model.con2 = pyo.Constraint(expr=2 * model.x_1 + 5 * model.x_2 >= 2)
