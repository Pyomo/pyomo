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

#
# Symmetric TSP with points on a plane
#
import re
import math
import pyomo.environ as pyo
from sc import pyomo_callback


def pyomo_create_model(options=None, model_options=None):
    model = pyo.ConcreteModel()
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
    model.N = pyo.Param(within=pyo.PositiveIntegers, initialize=N)
    #
    # Index set for points
    model.POINTS = pyo.RangeSet(1, model.N)

    #
    # (x,y) location
    def x_rule(model, i):
        return x[i - 1]

    model.x = pyo.Param(model.POINTS)

    def y_rule(model, i):
        return y[i - 1]

    model.y = pyo.Param(model.POINTS)

    #
    # Derived data
    #
    #
    # All points are connected
    def LINKS_rule(model):
        return set([(i, j) for i in model.POINTS for j in model.POINTS if i < j])

    model.LINKS = pyo.Set(dimen=2)

    #
    # Distance between points
    def d_rule(model, i, j):
        return math.sqrt(
            (model.x[i] - model.x[j]) ** 2 + (model.y[i] - model.y[j]) ** 2
        )

    model.d = pyo.Param(model.LINKS)
    #
    # Variables
    #
    model.Z = pyo.Var(model.LINKS, within=pyo.Binary)
    #
    # Constraints
    #
    #
    # Minimize tour length
    model.tour_length = pyo.Objective(expr=pyo.sum_product(model.d, model.Z))

    #
    # Each vertex has degree 2
    def Degrees_rule(model, i):
        return (
            sum(model.Z[i, j] for (i_, j) in model.LINKS if i == i_)
            + sum(model.Z[j, i] for (j, i_) in model.LINKS if i == i_)
            == 2
        )

    model.Degrees = pyo.Constraint(model.POINTS)

    #
    # NOTE: subtour constraints are added dynamically
    #
    ## Number of subtour elimination cuts
    # model.M = pyo.Param(within=pyo.PositiveIntegers)
    ## Number of subtour elimination cuts appended
    # model.numcut = pyo.Param(within=pyo.NonNegativeIntegers)
    # model.CUTINDICES = pyo.RangeSet(1,model.M)
    # model.CUTSET = pyo.Set(model.CUTINDICES)
    return model


@pyomo_callback('cut-callback')
def cut_callback(solver, model):
    print("Adding cuts")
