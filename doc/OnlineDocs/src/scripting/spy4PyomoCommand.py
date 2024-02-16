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
Code snippets for PyomoCommand.rst in testable form
"""

from pyomo.environ import *

model = ConcreteModel()
model.I = RangeSet(3)
model.J = RangeSet(3)
model.a = Param(model.I, model.J, default=1.0)
model.x = Var(model.J)
model.b = Param(model.I, default=1.0)


# @Troubleshooting_printed_command
def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    print("ax_constraint_rule was called for i=", str(i))
    return sum(model.a[i, j] * model.x[j] for j in model.J) >= model.b[i]


# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = Constraint(model.I, rule=ax_constraint_rule)
# @Troubleshooting_printed_command
