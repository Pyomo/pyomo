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
import math
import itertools
from functools import cmp_to_key


class Triangulation:
    Delaunay = 1
    J1 = 2

def get_j1_triangulation(points, dimension):
    points_map, K = _process_points_j1(points, dimension)
    if dimension == 2:
        return _get_j1_triangulation_2d(points_map, K)
    elif dimension == 3:
        return _get_j1_triangulation_3d(points, dimension)
    else:
        return _get_j1_triangulation_for_more_than_4d(points, dimension)


# Does some validation but mostly assumes the user did the right thing
def _process_points_j1(points, dimension):
    if not len(points[0]) == dimension:
        raise ValueError("Points not consistent with specified dimension")
    K = math.floor(len(points) ** (1 / dimension))
    if not len(points) == K**dimension:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    if not K % 2 == 1:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    
    # munge the points into an organized map with n-dimensional keys
    points.sort(key=cmp_to_key(_compare_lexicographic(dimension)))
    points_map = {}
    for point_index in itertools.product(range(K), repeat=dimension):
        point_flat_index = 0
        for n in range(dimension):
            point_flat_index += point_index[dimension - 1 - n] * K**n
        points_map[point_index] = points[point_flat_index]
    return points_map, K

def _compare_lexicographic(dimension):
    def compare_lexicographic_real(x, y):
        for n in range(dimension):
            if x[n] < y[n]:
                return -1
            elif y[n] < x[n]:
                return 1
        return 0
    return compare_lexicographic_real

def _get_j1_triangulation_2d(points_map, K):
    # Each square needs two triangles in it, orientation determined by the parity of
    # the bottom-left corner's coordinate indices (x and y). Same parity = top-left
    # and bottom-right triangles; different parity = top-right and bottom-left triangles.
    simplices = []
    for i in range(K):
        for j in range(K):
            if i % 2 == j % 2:
                simplices.append(
                    (points_map[i, j], 
                    points_map[i + 1, j + 1], 
                    points_map[i, j + 1]))
                simplices.append(
                    (points_map[i, j], 
                    points_map[i + 1, j + 1], 
                    points_map[i + 1, j]))
            else:
                simplices.append(
                    (points_map[i + 1, j], 
                    points_map[i, j + 1], 
                    points_map[i, j]))
                simplices.append(
                    (points_map[i + 1, j], 
                    points_map[i, j + 1], 
                    points_map[i + 1, j + 1]))
    return simplices

def _get_j1_triangulation_3d(points, dimension):
    pass

def _get_j1_triangulation_for_more_than_4d(points, dimension):
    pass
