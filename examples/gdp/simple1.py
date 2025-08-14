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

# Example: modeling a complementarity condition as a
#   disjunction
#
# This model does not work with existing transformations.
# See simple2.py and simple3.py for variants that work.

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction


def build_model():
    model = pyo.ConcreteModel()

    # x >= 0 _|_ y>=0
    model.x = pyo.Var(bounds=(0, None))
    model.y = pyo.Var(bounds=(0, None))

    # Two conditions
    def _d(disjunct, flag):
        model = disjunct.model()
        if flag:
            # x == 0
            disjunct.c = pyo.Constraint(expr=model.x == 0)
        else:
            # y == 0
            disjunct.c = pyo.Constraint(expr=model.y == 0)

    model.d = Disjunct([0, 1], rule=_d)

    # Define the disjunction
    def _c(model):
        return [model.d[0], model.d[1]]

    model.c = Disjunction(rule=_c)

    model.C = pyo.Constraint(expr=model.x + model.y <= 1)

    model.o = pyo.Objective(expr=2 * model.x + 3 * model.y, sense=pyo.maximize)
    return model
