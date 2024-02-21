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

#
# Symmetric TSP with points on a plane
#
import re
import math
from pyomo.core import *


def pyomo_create_model(options=None, model_options=None):
    model = ConcreteModel()
    #
    # Read from the 'points' file
    #
    filename = 'points' if options.points is None else options.points
    INPUT = open(filename, 'r')
    N = int(INPUT.readline())
    x = []
    y = []
    for line in INPUT:
        line = line.strip()
        tokens = re.split('[ /t]+', line)
        x.append(int(tokens[1]))
        y.append(int(tokens[2]))
    INPUT.close()
    #
    # Data initialized from file
    #
    #
    # Number of points
    model.N = Param(within=PositiveIntegers, initialize=N)
    #
    # Index set for points
    model.POINTS = RangeSet(1, model.N)

    #
    # (x,y) location
    def x_rule(model, i):
        return x[i - 1]

    model.x = Param(model.POINTS)

    def y_rule(model, i):
        return y[i - 1]

    model.y = Param(model.POINTS)

    #
    # Derived data
    #
    #
    # All points are connected
    def LINKS_rule(model):
        return set([(i, j) for i in model.POINTS for j in model.POINTS if i < j])

    model.LINKS = Set(dimen=2)

    #
    # Distance between points
    def d_rule(model, i, j):
        return math.sqrt(
            (model.x[i] - model.x[j]) ** 2 + (model.y[i] - model.y[j]) ** 2
        )

    model.d = Param(model.LINKS)
    #
    # Variables
    #
    model.Z = Var(model.LINKS, within=Binary)
    #
    # Constraints
    #
    #
    # Minimize tour length
    model.tour_length = Objective(expr=sum_product(model.d, model.Z))

    #
    # Each vertex has degree 2
    def Degrees_rule(model, i):
        return (
            sum(model.Z[i, j] for (i_, j) in model.LINKS if i == i_)
            + sum(model.Z[j, i] for (j, i_) in model.LINKS if i == i_)
            == 2
        )

    model.Degrees = Constraint(model.POINTS)

    #
    # NOTE: subtour constraints are added dynamically
    #
    ## Number of subtour elimination cuts
    # model.M = Param(within=PositiveIntegers)
    ## Number of subtour elimination cuts appended
    # model.numcut = Param(within=NonNegativeIntegers)
    # model.CUTINDICES = RangeSet(1,model.M)
    # model.CUTSET = Set(model.CUTINDICES)
    return model


@pyomo_callback('cut-callback')
def cut_callback(solver, model):
    print("Adding cuts")
