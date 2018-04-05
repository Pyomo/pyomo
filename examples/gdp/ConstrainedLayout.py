from __future__ import division
import pyomo.environ
from pyomo.core import *
from pyomo.gdp import *

# 2-D Constrained layout example from http://minlp.org/library/lib.php?lib=GDP

# Given a set of boxes and a set of circular areas, we determine a layout for
# the boxes which minimizes the distance among them (according to various costs
# associated with each pair of boxes) while ensuring that each box is within at
# least one of the circular areas

model = AbstractModel()

# Have to specify M since this is nonlinear
model.BigM = Suffix(direction=Suffix.LOCAL)
model.BigM[None] = 7000

model.BOXES = Set()
model.CIRCLES = Set()

model.Height = Param(model.BOXES)
model.Length = Param(model.BOXES)

model.Radius = Param(model.CIRCLES)
model.CenterXCoord = Param(model.CIRCLES)
model.CenterYCoord = Param(model.CIRCLES)

# generate the list of possible pairs of boxes
def box_pairs_filter(model, i, j):
    return i < j
model.BoxPairs = Set(initialize=model.BOXES * model.BOXES, dimen=2,
                     filter=box_pairs_filter)

# bounds for box coordinates
# TODO: What are they trying to accomplish here? Why is it always index
# 2 for the first two??
def max_x_coord_init(model, i):
    return model.CenterXCoord[2] + model.Radius[2] - model.Length[i]/2
model.XCoordUB = Param(model.BOXES, initialize=max_x_coord_init)

def max_y_coord_init(model, i):
    return model.CenterYCoord[2] + model.Radius[2] - model.Height[i]/2 
model.YCoordUB = Param(model.BOXES, initialize=max_y_coord_init)

def min_x_coord_init(model, i):
    return model.CenterXCoord[1] - model.Radius[1] + model.Length[i]/2
model.XCoordLB = Param(model.BOXES, initialize=min_x_coord_init)

def min_y_coord_init(model, i):
    return model.CenterYCoord[1] - model.Radius[1] + model.Height[i]/2
model.YCoordLB = Param(model.BOXES, initialize=min_y_coord_init)

# Cost associated with the distance between box i and j
model.DistCost = Param(model.BoxPairs, default=0)

model.BoxRelations = Set(initialize=['LeftOf', 'RightOf', 'Above', 'Below'])


## Variables #########

def get_x_bounds(model, i):
    return (model.XCoordLB[i], model.XCoordUB[i])
model.xCoord = Var(model.BOXES, bounds=get_x_bounds)
def get_y_bounds(model, i):
    return (model.YCoordLB[i], model.YCoordUB[i])
model.yCoord = Var(model.BOXES, bounds=get_y_bounds)

model.horizDist = Var(model.BoxPairs, within=NonNegativeReals)
model.vertDist = Var(model.BoxPairs, within=NonNegativeReals)


## Constraints #########

# Objective: Minimize relative distance
def rel_dist_rule(model):
    return sum(model.DistCost[i,j] * (model.horizDist[i,j] + model.vertDist[i,j])
               for (i,j) in model.BoxPairs)
model.rel_dist = Objective(rule=rel_dist_rule)


def horiz_dist_rule1(model, i, j):
    return model.horizDist[i,j] >= model.xCoord[i] - model.xCoord[j]
model.horiz_dist1 = Constraint(model.BoxPairs, rule=horiz_dist_rule1)

def horiz_dist_rule2(model, i, j):
    return model.horizDist[i,j] >= model.xCoord[j] - model.xCoord[i]
model.horiz_dist2 = Constraint(model.BoxPairs, rule=horiz_dist_rule2)

def vert_dist_rule1(model, i, j):
    return model.vertDist[i,j] >= model.yCoord[i] - model.yCoord[j]
model.vert_dist1 = Constraint(model.BoxPairs, rule=vert_dist_rule1)

def vert_dist_rule2(model, i, j):
    return model.vertDist[i,j] >= model.yCoord[j] - model.yCoord[i]
model.vert_dist2 = Constraint(model.BoxPairs, rule=vert_dist_rule2)


def no_overlap_disjunct_rule(disjunct, i, j, boxRelation):
    model = disjunct.model()   
    if boxRelation == 'LeftOf':
        disjunct.c = Constraint(expr=model.xCoord[i] + model.Length[i]/2 <= \
                                model.xCoord[j] - model.Length[j]/2)
    elif boxRelation == 'RightOf':
        disjunct.c = Constraint(expr=model.xCoord[i] - model.Length[i]/2 >= \
                                model.xCoord[j] + model.Length[j]/2)
    elif boxRelation == 'Above':
        disjunct.c = Constraint(expr=model.yCoord[i] - model.Height[i]/2 >= \
                                model.yCoord[j] + model.Height[j]/2)
    elif boxRelation == 'Below':
        disjunct.c = Constraint(expr=model.yCoord[i] + model.Height[i]/2 <= \
                                model.yCoord[j] - model.Height[j]/2)
    else:
        raise RuntimeError("Unrecognized box relationship: %s" % boxRelation)
model.no_overlap_disjunct = Disjunct(model.BoxPairs, model.BoxRelations,
    rule=no_overlap_disjunct_rule)

def no_overlap_rule(model, i, j):
    return [model.no_overlap_disjunct[i, j, direction] for direction in \
            model.BoxRelations]
model.no_overlap_disjunction = Disjunction(model.BoxPairs, rule=no_overlap_rule)

def in_circ_disjunct_rule(disjunct, i, k):
    model = disjunct.model()
    if 1:
        disjunct.upperLeftIn = Constraint(
            expr=(model.xCoord[i] - model.Length[i]/2 - \
                  model.CenterXCoord[k])**2 + \
            (model.yCoord[i] + model.Height[i]/2 - model.CenterYCoord[k])**2 <= \
            model.Radius[k]**2)
        disjunct.lowerLeftIn = Constraint(
            expr=(model.xCoord[i] - model.Length[i]/2 - \
                  model.CenterXCoord[k])**2 + \
            (model.yCoord[i] - model.Height[i]/2 - model.CenterYCoord[k])**2 <= \
            model.Radius[k]**2)
        disjunct.upperRightIn = Constraint(
            expr=(model.xCoord[i] + model.Length[i]/2 - \
                  model.CenterXCoord[k])**2 + \
            (model.yCoord[i] + model.Height[i]/2 - model.CenterYCoord[k])**2 <= \
            model.Radius[k]**2)
        disjunct.lowerRightIn = Constraint(
            expr=(model.xCoord[i] + model.Length[i]/2 - \
                  model.CenterXCoord[k])**2 + \
            (model.yCoord[i] - model.Height[i]/2 - model.CenterYCoord[k])**2 <= \
            model.Radius[k]**2)
model.in_circ_disjunct = Disjunct(model.BOXES, model.CIRCLES,
                                  rule=in_circ_disjunct_rule)

def in_circ_rule(model, i):
    return [model.in_circ_disjunct[i, k] for k in model.CIRCLES]
model.in_circ_disjunction = Disjunction(model.BOXES, rule=in_circ_rule)
