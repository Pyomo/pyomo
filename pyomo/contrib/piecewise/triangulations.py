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

from pytest import set_trace


class Triangulation:
    Delaunay = 1
    J1 = 2

def get_j1_triangulation(points, dimension):
    if dimension == 2:
        return _get_j1_triangulation_2d(points, dimension)
    elif dimension == 3:
        return _get_j1_triangulation_3d(points, dimension)
    else:
        return _get_j1_triangulation_for_more_than_4d(points, dimension)

def _get_j1_triangulation_2d(points, dimension):
    # I think this means coding up the proof by picture...
    pass

def _get_j1_triangulation_3d(points, dimension):
    pass

def _get_j1_triangulation_for_more_than_4d(points, dimension):
    pass
