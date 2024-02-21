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

"""
David L. Woodruff and Mingye Yang, Spring 2018
Code snippets for Constraints.rst in testable form
"""

from pyomo.environ import *

model = ConcreteModel()
# @Inequality_constraints_2expressions
model.x = Var()


def aRule(model):
    return model.x >= 2


model.Boundx = Constraint(rule=aRule)


def bRule(model):
    return (2, model.x, None)


model.boundx = Constraint(rule=bRule)
# @Inequality_constraints_2expressions

model = ConcreteModel()
model.J = Set(initialize=['butter', 'scones'])
model.x = Var(model.J)


# @Constraint_example
def teaOKrule(model):
    return model.x['butter'] + model.x['scones'] == 3


model.TeaConst = Constraint(rule=teaOKrule)
# @Constraint_example

# @Passing_elements_crossproduct
model.A = RangeSet(1, 10)
model.a = Param(model.A, within=PositiveReals)
model.ToBuy = Var(model.A)


def bud_rule(model, i):
    return model.a[i] * model.ToBuy[i] <= i


aBudget = Constraint(model.A, rule=bud_rule)
# @Passing_elements_crossproduct
