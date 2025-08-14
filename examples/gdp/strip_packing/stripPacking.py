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

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

# Strip-packing example from http://minlp.org/library/lib.php?lib=GDP

# This model packs a set of rectangles without rotation or overlap within a
# strip of a given width, minimizing the length of the strip.

model = pyo.AbstractModel()

model.RECTANGLES = pyo.Set(ordered=True)

# height and width of each rectangle
model.Heights = pyo.Param(model.RECTANGLES)
model.Lengths = pyo.Param(model.RECTANGLES)

# width of strip
model.StripWidth = pyo.Param()


# upperbound on length (default is sum of lengths of rectangles)
def sumLengths(model):
    return sum(model.Lengths[i] for i in model.RECTANGLES)


model.LengthUB = pyo.Param(initialize=sumLengths)

# rectangle relations
model.RecRelations = pyo.Set(initialize=['LeftOf', 'RightOf', 'Above', 'Below'])

# x and y coordinates of each of the rectangles
model.x = pyo.Var(model.RECTANGLES, bounds=(0, model.LengthUB))
model.y = pyo.Var(model.RECTANGLES, bounds=(0, model.StripWidth))

# length of strip (this will be the objective)
model.Lt = pyo.Var(within=pyo.NonNegativeReals)


# generate the list of possible rectangle conflicts (which are any pair)
def rec_pairs_filter(model, i, j):
    return i < j


model.RectanglePairs = pyo.Set(
    initialize=model.RECTANGLES * model.RECTANGLES, dimen=2, filter=rec_pairs_filter
)


# strip length constraint
def strip_ends_after_last_rec_rule(model, i):
    return model.Lt >= model.x[i] + model.Lengths[i]


model.strip_ends_after_last_rec = pyo.Constraint(
    model.RECTANGLES, rule=strip_ends_after_last_rec_rule
)


# constraints to prevent rectangles from going off strip
def no_recs_off_end_rule(model, i):
    return pyo.inequality(0, model.x[i], model.LengthUB - model.Lengths[i])


model.no_recs_off_end = pyo.Constraint(model.RECTANGLES, rule=no_recs_off_end_rule)


def no_recs_off_bottom_rule(model, i):
    return pyo.inequality(model.Heights[i], model.y[i], model.StripWidth)


model.no_recs_off_bottom = pyo.Constraint(
    model.RECTANGLES, rule=no_recs_off_bottom_rule
)


# Disjunctions to prevent overlap between rectangles
def no_overlap_disjunct_rule(disjunct, i, j, recRelation):
    model = disjunct.model()
    # left
    if recRelation == 'LeftOf':
        disjunct.c = pyo.Constraint(expr=model.x[i] + model.Lengths[i] <= model.x[j])
    # right
    elif recRelation == 'RightOf':
        disjunct.c = pyo.Constraint(expr=model.x[j] + model.Lengths[j] <= model.x[i])
    # above
    elif recRelation == 'Above':
        disjunct.c = pyo.Constraint(expr=model.y[i] - model.Heights[i] >= model.y[j])
    # below
    elif recRelation == 'Below':
        disjunct.c = pyo.Constraint(expr=model.y[j] - model.Heights[j] >= model.y[i])
    else:
        raise RuntimeError("Unrecognized rectangle relationship: %s" % recRelation)


model.no_overlap_disjunct = Disjunct(
    model.RectanglePairs, model.RecRelations, rule=no_overlap_disjunct_rule
)


def no_overlap(model, i, j):
    return [
        model.no_overlap_disjunct[i, j, direction] for direction in model.RecRelations
    ]


model.disj = Disjunction(model.RectanglePairs, rule=no_overlap)

# minimize length
model.length = pyo.Objective(expr=model.Lt)
