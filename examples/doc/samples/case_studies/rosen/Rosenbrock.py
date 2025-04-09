#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# @intro:
import pyomo.environ as pyo

model = pyo.AbstractModel()
# @:intro
# @vars:
model.x = pyo.Var(initialize=1.5)
model.y = pyo.Var(initialize=1.5)


# @:vars
# @obj:
def rosenbrock(amodel):
    return (1.0 - amodel.x) ** 2 + 100.0 * (amodel.y - amodel.x**2) ** 2


model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
# @:obj
