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
Code snippets for Variables.rst in testable form
"""

from pyomo.environ import *

model = ConcreteModel()
# @Declare_singleton_variable
model.LumberJack = Var(within=NonNegativeReals, bounds=(0, 6), initialize=1.5)
# @Declare_singleton_variable

# @Assign_value
model.LumberJack = 1.5
# @Assign_value

# @Declare_bounds
model.A = Set(initialize=['Scones', 'Tea'])
lb = {'Scones': 2, 'Tea': 4}
ub = {'Scones': 5, 'Tea': 7}


def fb(model, i):
    return (lb[i], ub[i])


model.PriceToCharge = Var(model.A, domain=PositiveIntegers, bounds=fb)
# @Declare_bounds
