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
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    Binary,
    Constraint,
    Param,
    SolverFactory,
    value,
    Objective,
    TerminationCondition,
)

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
    #points.sort(key=cmp_to_key(_compare_lexicographic(dimension)))
    # verify: does this do correct sorting by default?
    points.sort()
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

def get_incremental_simplex_ordering(simplices, subsolver='gurobi'):
    # Set up a MIP (err, MIQCP) that orders our simplices and their vertices for us
    # in the following way:
    #
    # (1) The simplices are ordered T_1, ..., T_N such that T_i has nonempty intersection
    #     with T_{i+1}. It doesn't have to be a whole face; just a vertex is enough.
    # (2) On each simplex T_i, the vertices are ordered T_i^1, ..., T_i^n such
    #     that T_i^n = T_{i+1}^1
    m = ConcreteModel()

    # Sets and Params
    m.SimplicesCount = Param(value=len(simplices))
    m.SIMPLICES = RangeSet(0, m.SimplicesCount - 1)
    # For each of the simplices we need to choose an initial and a final vertex.
    # The rest we can order arbitrarily after finishing the MIP solve.
    m.SimplexVerticesCount = Param(value=len(simplices[0]))
    m.VERTEX_INDICES = RangeSet(0, m.SimplexVerticesCount - 1)
    @m.Param(m.SIMPLICES, m.VERTEX_INDICES, m.SIMPLICES, m.VERTEX_INDICES, domain=Binary)
    def TestVerticesEqual(m, i, n, j, k):
        return 1 if simplices[i][n] == simplices[j][k] else 0

    # Vars
    # x_ij means simplex i is placed in slot j
    m.x = Var(m.SIMPLICES, m.SIMPLICES, domain=Binary)
    m.vertex_is_first = Var(m.SIMPLICES, m.VERTEX_INDICES, domain=Binary)
    m.vertex_is_last = Var(m.SIMPLICES, m.VERTEX_INDICES, domain=Binary)


    # Constraints
    # Each simplex should have a slot and each slot should have a simplex
    @m.Constraint(m.SIMPLICES)
    def schedule_each_simplex(m, i):
        return sum(m.x[i, j] for j in m.SIMPLICES) == 1
    @m.Constraint(m.SIMPLICES)
    def schedule_each_slot(m, j):
        return sum(m.x[i, j] for i in m.SIMPLICES) == 1
    
    # Enforce property (1)
    @m.Constraint(m.SIMPLICES)
    def simplex_order(m, i):
        if i == m.SimplicesCount - 1:
            return Constraint.Skip # no ordering for the last one
        # anything with at least a vertex in common is a neighbor
        neighbors = [s for s in m.SIMPLICES if sum(TestVerticesEqual[i, n, s, k] for n in m.VERTEX_INDICES for k in m.VERTEX_INDICES) >= 1]
        return sum(m.x[i, j] * m.x[k, j+1] for j in m.SIMPLICES for k in neighbors) == 1

    # Each simplex needs exactly one first and exactly one last vertex
    @m.Constraint(m.SIMPLICES)
    def one_first_vertex(m, i):
        return sum(m.vertex_is_first[i, n] for n in m.VERTEX_INDICES) == 1
    @m.Constraint(m.SIMPLICES)
    def one_last_vertex(m, i):
        return sum(m.vertex_is_last[i, n] for n in m.VERTEX_INDICES) == 1
    
    # Enforce property (2)
    @m.Constraint(m.SIMPLICES, m.SIMPLICES)
    def vertex_order(m, i, j):
        if i == m.SimplicesCount - 1:
            return Constraint.Skip # no ordering for the last one
        # Enforce only when j is the simplex following i. If not, RHS is zero
        return (
            sum(m.vertex_is_last[i, n] * m.vertex_is_first[j, k] * m.TestVerticesEqual[i, n, j, k] for n in m.VERTEX_INDICES for k in m.VERTEX_INDICES) 
            >= sum(m.x[i, p] * m.x[j, p + 1] for p in m.SIMPLICES if p != m.SimplicesCount - 1)
        )
    
    # Trivial objective (do I need this?)
    m.obj = Objective(expr=0)
    
    # Solve model
    results = SolverFactory(subsolver).solve(m)
    match(results.solver.termination_condition):
        case TerminationCondition.infeasible:
            raise ValueError("The triangulation was impossible to suitably order for the incremental transformation. Try a different triangulation, such as J1.")
        case TerminationCondition.feasible:
            pass
        case _:
            raise ValueError(f"Failed to generate suitable ordering for incremental transformation due to unexpected termination condition {results.solver.termination_condition}")
    
    # Retrieve data
    simplex_ordering = {}
    for i in m.SIMPLICES:
        for j in m.SIMPLICES:
            if abs(value(m.x[i, j]) - 1) < 1e-5:
                simplex_ordering[i] = j
                break
    vertex_ordering = {}
    for i in m.SIMPLICES:
        first = None
        last = None
        for n in m.VERTEX_INDICES:
            if abs(value(m.vertex_is_first[i, n]) - 1) < 1e-5:
                first = n
                vertex_ordering[i, 0] = first
            if abs(value(m.vertex_is_last[i, n]) - 1) < 1e-5:
                last = n
                vertex_ordering[i, m.SimplexVerticesCount - 1] = last
            if first is not None and last is not None:
                break
        # Fill in the middle ones arbitrarily
        idx = 1
        for j in range(m.SimplexVerticesCount):
            if j != first and j != last:
                vertex_ordering[idx] = j
                idx += 1

    return simplex_ordering, vertex_ordering
