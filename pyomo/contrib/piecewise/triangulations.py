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
from types import SimpleNamespace
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
    points_map, num_pts = _process_points_j1(points, dimension)
    simplices_list = _get_j1_triangulation(points_map, num_pts - 1, dimension)
    # make a duck-typed thing that superficially looks like an instance of 
    # scipy.spatial.Delaunay (these are NDarrays in the original)
    triangulation = SimpleNamespace()
    triangulation.points = list(range(len(simplices_list)))
    triangulation.simplices = {i: simplices_list[i] for i in triangulation.points}
    triangulation.coplanar = []

    return triangulation

    #if dimension == 2:
    #    return _get_j1_triangulation_2d(points_map, num_pts)
    #elif dimension == 3:
    #    return _get_j1_triangulation_3d(points, dimension)
    #else:
    #    return _get_j1_triangulation_for_more_than_4d(points, dimension)


# Does some validation but mostly assumes the user did the right thing
def _process_points_j1(points, dimension):
    if not len(points[0]) == dimension:
        raise ValueError("Points not consistent with specified dimension")
    num_pts = math.floor(len(points) ** (1 / dimension))
    if not len(points) == num_pts**dimension:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    if not num_pts % 2 == 1:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    
    # munge the points into an organized map with n-dimensional keys
    #points.sort(key=cmp_to_key(_compare_lexicographic(dimension)))
    # verify: does this do correct sorting by default?
    points.sort()
    points_map = {}
    for point_index in itertools.product(range(num_pts), repeat=dimension):
        point_flat_index = 0
        for n in range(dimension):
            point_flat_index += point_index[dimension - 1 - n] * num_pts**n
        points_map[point_index] = points[point_flat_index]
    return points_map, num_pts

#def _compare_lexicographic(dimension):
#    def compare_lexicographic_real(x, y):
#        for n in range(dimension):
#            if x[n] < y[n]:
#                return -1
#            elif y[n] < x[n]:
#                return 1
#        return 0
#    return compare_lexicographic_real

# This implements the J1 "Union Jack" triangulation (Todd 77) as explained by
# Vielma 2010.
# Triangulate {0, ..., K}^n for even K using the J1 triangulation, mapping the
# obtained simplices through the points_map for a slight generalization.
def _get_j1_triangulation(points_map, K, n):
    if K % 2 != 0:
        raise ValueError("K must be even")
    # 1, 3, ..., K - 1
    axis_odds = range(1, K, 2)
    V_0 = itertools.product(axis_odds, repeat=n)
    big_iterator = itertools.product(V_0, 
                                     itertools.permutations(range(0, n), n), 
                                     itertools.product((-1, 1), repeat=n))
    ret = []
    for v_0, pi, s in big_iterator:
        simplex = []
        current = list(v_0)
        simplex.append(points_map[*current])
        for i in range(0, n):
            current = current.copy()
            current[pi[i]] += s[pi[i]]
            simplex.append(points_map[*current])
        # sort this because it might happen again later and we'd like to stay
        # consistent. Undo this if it's slow.
        ret.append(sorted(simplex)) 
    return ret

def _get_j1_triangulation_2d(points_map, num_pts):
    # Each square needs two triangles in it, orientation determined by the parity of
    # the bottom-left corner's coordinate indices (x and y). Same parity = top-left
    # and bottom-right triangles; different parity = top-right and bottom-left triangles.
    simplices = []
    for i in range(num_pts):
        for j in range(num_pts):
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
    # 
    # Note that (2) implies (1), so we only need to enforce that.
    #
    # TODO: issue: I don't think gurobi is magical enough to notice the special structure
    # of this so it's basically looking for a hamiltonian path in a big graph...
    # If we want to resolve this, we need to at least partially go back to the 70s thing
    #
    # An alternative approach is to order the simplices instead of the vertices. To
    # do this, the condition (1) should be that they share a 1-face, not just a
    # vertex. Then there is always a consistent way to choose distinct first and
    # last vertices, which would otherwise be the issue - the rest of the vertex
    # ordering can be arbitrary. Then we are really looking for a hamiltonian
    # path which is what Todd did. However, we then fail to find orderings for
    # strange triangulations such as two triangles intersecting at a point.
    m = ConcreteModel()

    # Sets and Params
    m.SimplicesCount = Param(initialize=len(simplices))
    m.SIMPLICES = RangeSet(0, m.SimplicesCount - 1)
    # For each of the simplices we need to choose an initial and a final vertex.
    # The rest we can order arbitrarily after finishing the MIP solve.
    m.SimplexVerticesCount = Param(initialize=len(simplices[0]))
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
    
    # Enforce property (1), but this is guaranteed by (2) so unnecessary
    #@m.Constraint(m.SIMPLICES)
    #def simplex_order(m, i):
    #    # anything with at least a vertex in common is a neighbor
    #    neighbors = [s for s in m.SIMPLICES if sum(m.TestVerticesEqual[i, n, s, k] for n in m.VERTEX_INDICES for k in m.VERTEX_INDICES) >= 1]
    #    return sum(m.x[i, j] * m.x[k, j+1] for j in m.SIMPLICES if j != m.SimplicesCount - 1 for k in neighbors) == 1

    # Each simplex needs exactly one first and exactly one last vertex
    @m.Constraint(m.SIMPLICES)
    def one_first_vertex(m, i):
        return sum(m.vertex_is_first[i, n] for n in m.VERTEX_INDICES) == 1
    @m.Constraint(m.SIMPLICES)
    def one_last_vertex(m, i):
        return sum(m.vertex_is_last[i, n] for n in m.VERTEX_INDICES) == 1
    
    # The last vertex cannot be the same as the first vertex
    @m.Constraint(m.SIMPLICES, m.VERTEX_INDICES)
    def first_last_distinct(m, i, n):
        return m.vertex_is_first[i, n] * m.vertex_is_last[i, n] == 0
    
    # Enforce property (2). This also guarantees property (1)
    @m.Constraint(m.SIMPLICES, m.SIMPLICES)
    def vertex_order(m, i, j):
        # Enforce only when j is the simplex following i. If not, RHS is zero
        return (
            sum(m.vertex_is_last[i, n] * m.vertex_is_first[j, k] * m.TestVerticesEqual[i, n, j, k] for n in m.VERTEX_INDICES for k in m.VERTEX_INDICES) 
            >= sum(m.x[i, p] * m.x[j, p + 1] for p in m.SIMPLICES if p != m.SimplicesCount - 1)
        )
    
    # Trivial objective (do I need this?)
    m.obj = Objective(expr=0)
    
    # Solve model
    results = SolverFactory(subsolver).solve(m, tee=True)
    match(results.solver.termination_condition):
        case TerminationCondition.infeasible:
            raise ValueError("The triangulation was impossible to suitably order for the incremental transformation. Try a different triangulation, such as J1.")
        case TerminationCondition.optimal:
            pass
        case _:
            raise ValueError(f"Failed to generate suitable ordering for incremental transformation due to unexpected solver termination condition {results.solver.termination_condition}")
    
    # Retrieve data
    #m.pprint()
    new_simplices = {}
    for j in m.SIMPLICES:
        for i in m.SIMPLICES:
            if abs(value(m.x[i, j]) - 1) < 1e-5:
                # The jth slot is occupied by the ith simplex
                old_simplex = simplices[i]
                # Reorder its vertices, too
                first = None
                last = None
                for n in m.VERTEX_INDICES:
                    if abs(value(m.vertex_is_first[i, n]) - 1) < 1e-5:
                        first = n
                    if abs(value(m.vertex_is_last[i, n]) - 1) < 1e-5:
                        last = n
                    if first is not None and last is not None:
                        break
                new_simplex = [old_simplex[first]]
                for n in m.VERTEX_INDICES:
                    if n != first and n != last:
                        new_simplex.append(old_simplex[n])
                new_simplex.append(old_simplex[last])
                new_simplices[j] = new_simplex
                break
    return new_simplices