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
from enum import Enum
from functools import cmp_to_key
from pyomo.common.errors import DeveloperError
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
from pyomo.common.dependencies import attempt_import
nx, nx_available = attempt_import(
    'networkx', 'Networkx is required to calculate incremental ordering.'
)

class Triangulation:
    Delaunay = 1
    J1 = 2

def get_j1_triangulation(points, dimension, ordered=False):
    points_map, num_pts = _process_points_j1(points, dimension)

    if ordered and dimension == 2:
        simplices_list = _get_j1_triangulation_2d_ordered(points_map, num_pts - 1)
    else:
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
    num_pts = round(len(points) ** (1 / dimension))
    if not len(points) == num_pts**dimension:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    if not num_pts % 2 == 1:
        raise ValueError("'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis")
    
    # munge the points into an organized map with n-dimensional keys
    points.sort()
    points_map = {}
    for point_index in itertools.product(range(num_pts), repeat=dimension):
        point_flat_index = 0
        for n in range(dimension):
            point_flat_index += point_index[dimension - 1 - n] * num_pts**n
        points_map[point_index] = points[point_flat_index]
    return points_map, num_pts

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

# Implement proof-by-picture from Todd 1977. I do the reverse order he does
# and also keep the pictures slightly more regular to make things easier to
# implement.
def _get_j1_triangulation_2d_ordered(points_map, num_pts):
    # check when square has simplices in top-left and bottom-right
    square_parity_tlbr = lambda x, y: x % 2 == y % 2
    # check when we are in a "turnaround square" as seen in the picture
    is_turnaround = lambda x, y: x >= num_pts / 2 and y == (num_pts / 2) - 1
    class Direction(Enum):
        left = 0
        down = 1
        up = 2
        right = 3
    facing = None

    simplices = {}
    start_square = (num_pts - 1, (num_pts / 2) - 1)

    # make it easier to read what I'm doing
    def add_bottom_right():
        simplices[len(simplices)] = (points_map[x, y], points_map[x + 1, y], points_map[x + 1, y + 1])
    def add_top_right():
        simplices[len(simplices)] = (points_map[x, y + 1], points_map[x + 1, y], points_map[x + 1, y + 1])
    def add_bottom_left():
        simplices[len(simplices)] = (points_map[x, y], points_map[x, y + 1], points_map[x + 1, y])
    def add_top_left():
        simplices[len(simplices)] = (points_map[x, y], points_map[x, y + 1], points_map[x + 1, y + 1])


    # identify square by bottom-left corner
    x, y = start_square
    used_squares = set() # not used for the turnaround squares

    # depending on parity we will need to go either up or down to start
    if square_parity_tlbr(x, y):
        add_bottom_right()
        facing = Direction.down
        y -= 1
    else:
        add_top_right()
        facing = Direction.up
        y += 1
    
    # state machine
    while (True):
        match(facing):
            case Direction.left:
                if square_parity_tlbr(x, y):
                    add_bottom_right()
                    add_top_left()
                else:
                    add_top_right()
                    add_bottom_left()
                used_squares.add((x, y))
                if (x - 1, y) in used_squares or x == 0:
                    # can't keep going left so we need to go up or down depending
                    # on parity
                    if square_parity_tlbr(x, y):
                        y += 1
                        facing = Direction.up
                        continue
                    else:
                        y -= 1
                        facing = Direction.down
                        continue
                else:
                    x -= 1
                    continue
            case Direction.right:
                if is_turnaround(x, y):
                    # finished; this case should always eventually be reached
                    add_bottom_left()
                    fix_vertices_incremental_order(simplices)
                    return simplices
                else:
                    if square_parity_tlbr(x, y):
                        add_top_left()
                        add_bottom_right()
                    else:
                        add_bottom_left()
                        add_top_right()
                used_squares.add((x, y))
                if (x + 1, y) in used_squares or x == num_pts - 1:
                    # can't keep going right so we need to go up or down depending
                    # on parity
                    if square_parity_tlbr(x, y):
                        y -= 1
                        facing = Direction.down
                        continue
                    else:
                        y += 1
                        facing = Direction.up
                        continue
                else:
                    x += 1
                    continue
            case Direction.down:
                if is_turnaround(x, y):
                    # we are always in a TLBR square. Take the TL of this, the TR
                    # of the one on the left, and continue upwards one to the left
                    assert square_parity_tlbr(x, y), "uh oh"
                    add_top_left()
                    x -= 1
                    add_top_right()
                    y += 1
                    facing = Direction.up
                    continue
                else:
                    if square_parity_tlbr(x, y):
                        add_top_left()
                        add_bottom_right()
                    else:
                        add_top_right()
                        add_bottom_left()
                    used_squares.add((x, y))
                    if (x, y - 1) in used_squares or y == 0:
                        # can't keep going down so we need to turn depending
                        # on our parity
                        if square_parity_tlbr(x, y):
                            x += 1
                            facing = Direction.right
                            continue
                        else:
                            x -= 1
                            facing = Direction.left
                            continue
                    else:
                        y -= 1
                        continue
            case Direction.up:
                if is_turnaround(x, y):
                    # we are always in a non-TLBR square. Take the BL of this, the BR
                    # of the one on the left, and continue downwards one to the left
                    assert not square_parity_tlbr(x, y), "uh oh"
                    add_bottom_left()
                    x -= 1
                    add_bottom_right()
                    y -= 1
                    facing = Direction.down
                    continue
                else:
                    if square_parity_tlbr(x, y):
                        add_bottom_right()
                        add_top_left()
                    else:
                        add_bottom_left()
                        add_top_right()
                    used_squares.add((x, y))
                    if (x, y + 1) in used_squares or y == num_pts - 1:
                        # can't keep going up so we need to turn depending
                        # on our parity
                        if square_parity_tlbr(x, y):
                            x -= 1
                            facing = Direction.left
                            continue
                        else:
                            x += 1
                            facing = Direction.right
                            continue
                    else:
                        y += 1
                        continue


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

# If we have the assumption that our ordering is possible such that consecutively
# ordered simplices share at least a one-face, then getting an order for the
# simplices is enough to get one for the edges and we "just" need to find a 
# Hamiltonian path
def get_incremental_simplex_ordering_assume_connected_by_n_face(simplices, connected_face_dim, subsolver='gurobi'):
    if connected_face_dim == 0:
        return get_incremental_simplex_ordering(simplices)
    #if not nx_available:
    #    raise ImportError('Missing Networkx')
    #G = nx.Graph()
    #G.add_nodes_from(range(len(simplices)))
    #for i in range(len(simplices)):
    #    for j in range(i + 1, len(simplices)):
    #        if len(set(simplices[i]) & set(simplices[j])) >= n + 1:
    #            G.add_edge(i, j)
    
    # ask Gurobi again because networkx doesn't seem to have a general hamiltonian
    # path and I don't want to implement it myself

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
        # anything with at least a vertex in common is a neighbor
        neighbors = [s for s in m.SIMPLICES if sum(m.TestVerticesEqual[i, n, s, k] for n in m.VERTEX_INDICES for k in m.VERTEX_INDICES) >= connected_face_dim + 1 and s != i]
        #print(f'neighbors of {i} are {neighbors}')
        return sum(m.x[i, j] * m.x[k, j + 1] for j in m.SIMPLICES if j != m.SimplicesCount - 1 for k in neighbors) + m.x[i, m.SimplicesCount - 1] == 1
    
    # Trivial objective (do I need this?)
    m.obj = Objective(expr=0)
    
    #m.pprint()
    # Solve model
    results = SolverFactory(subsolver).solve(m, tee=True)
    match(results.solver.termination_condition):
        case TerminationCondition.infeasible:
            raise ValueError(f"The triangulation was impossible to suitably order for the incremental transformation under the assumption that consecutive simplices share {connected_face_dim}-faces. Try relaxing that assumption, or try a different triangulation, such as J1.")
        case TerminationCondition.optimal:
            pass
        case _:
            raise ValueError(f"Failed to generate suitable ordering for incremental transformation due to unexpected solver termination condition {results.solver.termination_condition}")

    # Retrieve data
    new_simplices = {}
    for j in m.SIMPLICES:
        for i in m.SIMPLICES:
            if abs(value(m.x[i, j]) - 1) < 1e-5:
                # The jth slot is occupied by the ith simplex
                new_simplices[j] = simplices[i]
                # Note vertices need to be fixed after the fact now
                break
    fix_vertices_incremental_order(new_simplices)
    return new_simplices

# Fix vertices (in place) when the simplices are right but vertices are not
def fix_vertices_incremental_order(simplices):
    last_vertex_index = len(simplices[0]) - 1
    for i, simplex in simplices.items():
        # Choose vertices like this: first is always the same as last
        # of the previous simplex. Last is arbitrarily chosen from the
        # intersection with the next simplex.
        first = None
        last = None
        if i == 0:
            first = 0
        else:
            for n in range(last_vertex_index + 1):
                if simplex[n] == simplices[i - 1][last_vertex_index]:
                    first = n
                    break
            
        if i == len(simplices) - 1:
            last = last_vertex_index
        else:
            for n in range(last_vertex_index + 1):
                if simplex[n] in simplices[i + 1] and n != first:
                    last = n
                    break
        if first == None or last == None:
            raise DeveloperError("Couldn't fix vertex ordering for incremental.")
        
        # reorder the simplex with the desired first and last
        new_simplex = [simplex[first]]
        for n in range(last_vertex_index + 1):
            if n != first and n != last:
                new_simplex.append(simplex[n])
        new_simplex.append(simplex[last])
        simplices[i] = new_simplex