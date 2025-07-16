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

"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Constraints.rst in testable form
"""

import pyomo.environ as pyo

model = pyo.ConcreteModel()
# @Inequality_constraints_2expressions
model.x = pyo.Var()


def aRule(model):
    return model.x >= 2


model.Boundx = pyo.Constraint(rule=aRule)


def bRule(model):
    return (2, model.x, None)


model.boundx = pyo.Constraint(rule=bRule)
# @Inequality_constraints_2expressions

model = pyo.ConcreteModel()
model.J = pyo.Set(initialize=['butter', 'scones'])
model.x = pyo.Var(model.J)


# @Constraint_example
def teaOKrule(model):
    return model.x['butter'] + model.x['scones'] == 3


model.TeaConst = pyo.Constraint(rule=teaOKrule)
# @Constraint_example

# @Passing_elements_crossproduct
model.A = pyo.RangeSet(1, 10)
model.a = pyo.Param(model.A, within=pyo.PositiveReals)
model.ToBuy = pyo.Var(model.A)


def bud_rule(model, i):
    return model.a[i] * model.ToBuy[i] <= i


aBudget = pyo.Constraint(model.A, rule=bud_rule)
# @Passing_elements_crossproduct
