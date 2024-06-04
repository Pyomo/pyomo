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

class Triangulation(Enum):
    AssumeValid = 0
    Delaunay = 1
    J1 = 2
    OrderedJ1 = 3


def get_unordered_j1_triangulation(points, dimension):
    points_map, num_pts = _process_points_j1(points, dimension)
    simplices_list = _get_j1_triangulation(points_map, num_pts - 1, dimension)
    # make a duck-typed thing that superficially looks like an instance of
    # scipy.spatial.Delaunay (these are NDarrays in the original)
    triangulation = SimpleNamespace()
    triangulation.points = list(range(len(simplices_list)))
    triangulation.simplices = {i: simplices_list[i] for i in triangulation.points}
    triangulation.coplanar = []

    return triangulation


def get_ordered_j1_triangulation(points, dimension):
    points_map, num_pts = _process_points_j1(points, dimension)
    if dimension == 2:
        simplices_list = _get_ordered_j1_triangulation_2d(points_map, num_pts - 1)
    elif dimension == 3:
        raise DeveloperError("Unimplemented!")
        #simplices_list = _get_ordered_j1_triangulation_3d(points_map, num_pts - 1)
    else:
        simplices_list = _get_ordered_j1_triangulation_4d_and_above(points_map, num_pts - 1, dimension)
    triangulation = SimpleNamespace()
    triangulation.points = list(range(len(simplices_list)))
    triangulation.simplices = {i: simplices_list[i] for i in triangulation.points}
    triangulation.coplanar = []

    return triangulation


# Does some validation but mostly assumes the user did the right thing
def _process_points_j1(points, dimension):
    if not len(points[0]) == dimension:
        raise ValueError("Points not consistent with specified dimension")
    num_pts = round(len(points) ** (1 / dimension))
    if not len(points) == num_pts**dimension:
        raise ValueError(
            "'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis"
        )
    if not num_pts % 2 == 1:
        raise ValueError(
            "'points' must have points forming an n-dimensional grid with straight grid lines and the same odd number of points in each axis"
        )

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
    big_iterator = itertools.product(
        V_0,
        itertools.permutations(range(0, n), n),
        itertools.product((-1, 1), repeat=n),
    )
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
# implement. Also remember that Todd's drawing is misleading to the point of
# almost being wrong so make sure you draw it properly first.
def _get_ordered_j1_triangulation_2d(points_map, num_pts):
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
        simplices[len(simplices)] = (
            points_map[x, y],
            points_map[x + 1, y],
            points_map[x + 1, y + 1],
        )

    def add_top_right():
        simplices[len(simplices)] = (
            points_map[x, y + 1],
            points_map[x + 1, y],
            points_map[x + 1, y + 1],
        )

    def add_bottom_left():
        simplices[len(simplices)] = (
            points_map[x, y],
            points_map[x, y + 1],
            points_map[x + 1, y],
        )

    def add_top_left():
        simplices[len(simplices)] = (
            points_map[x, y],
            points_map[x, y + 1],
            points_map[x + 1, y + 1],
        )

    # identify square by bottom-left corner
    x, y = start_square
    used_squares = set()  # not used for the turnaround squares

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
    while True:
        match (facing):
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


def _get_ordered_j1_triangulation_3d(points_map, num_pts):
    pass


def _get_ordered_j1_triangulation_4d_and_above(points_map, num_pts, dim):
    # step one: get a hamiltonian path in the appropriate grid graph (low-coordinate
    # corners of the grid squares)
    grid_hamiltonian = get_grid_hamiltonian(dim, num_pts)

    # step 1.5: get a starting simplex. Anything that is *not* adjacent to the
    # second square is fine. Since we always go from [0, ..., 0] to [0, ..., 1],
    # i.e., j=`dim`, anything where `dim` is not the first or last symbol should
    # always work. Let's stick it in the second place
    start_perm = tuple([1] + [dim] + list(range(2, dim)))

    # step two: for each square, get a sequence of simplices from a starting simplex,
    # through the square, and then ending with a simplex adjacent to the next square.
    # Then find the appropriate adjacent simplex to start on the next square
    simplices = {}
    for i in range(len(grid_hamiltonian) - 1):
        current_corner = grid_hamiltonian[i]
        next_corner = grid_hamiltonian[i + 1]
        # differing index
        j = [k + 1 for k in range(dim) if current_corner[k] != next_corner[k]][0]
        # border x_j value between this square and next
        c = max(current_corner[j - 1], next_corner[j - 1])
        v_0, sign = get_nearest_odd_and_sign_vec(current_corner)
        # According to Todd, what we need is to end with a permutation where rho(n) = j
        # if c is odd, and end with one where rho(1) = j if c is even. I think this
        # is right -- basically the sign from the sign vector sometimes cancels
        # out the sign from whether we are entering in the +c or -c direction.
        if c % 2 == 0:
            perm_sequence = get_Gn_hamiltonian(dim, start_perm, j, False)
            for pi in perm_sequence:
                simplices[len(simplices)] = get_one_j1_simplex(v_0, pi, sign, dim, points_map)
        else:
            perm_sequence = get_Gn_hamiltonian(dim, start_perm, j, True)
            for pi in perm_sequence:
                simplices[len(simplices)] = get_one_j1_simplex(v_0, pi, sign, dim, points_map)
        # should be true regardless of odd or even? I hope
        start_perm = perm_sequence[-1]

    # step three: finish out the last square
    # Any final permutation is fine; we are going nowhere after this
    v_0, sign = get_nearest_odd_and_sign_vec(grid_hamiltonian[-1])
    for pi in get_Gn_hamiltonian(dim, start_perm, 1, False):
        simplices[len(simplices)] = get_one_j1_simplex(v_0, pi, sign, dim, points_map)
    
    # fix vertices and return
    fix_vertices_incremental_order(simplices)
    return simplices

def get_one_j1_simplex(v_0, pi, sign, dim, points_map):
    simplex = []
    current = list(v_0)
    simplex.append(points_map[*current])
    for i in range(0, dim):
        current = current.copy()
        current[pi[i] - 1] += sign[pi[i] - 1]
        simplex.append(points_map[*current])
    return sorted(simplex)

# get the v_0 and sign vectors corresponding to a given square, identified by its
# low-coordinate corner
def get_nearest_odd_and_sign_vec(corner):
    v_0 = []
    sign = []
    for x in corner:
        if x % 2 == 0:
            v_0.append(x + 1)
            sign.append(-1)
        else:
            v_0.append(x)
            sign.append(1)
    return v_0, sign

def get_grid_hamiltonian(dim, length):
    if dim == 1:
        return [[n] for n in range(length)]
    else:
        ret = []
        prev = get_grid_hamiltonian(dim - 1, length)
        for n in range(length):
            # if n is even, add the previous hamiltonian with n in its new first
            # coordinate. If odd, do the same with the previous hamiltonian in reverse.
            if n % 2 == 0:
                for x in prev:
                    ret.append([n] + x)
            else:
                for x in reversed(prev):
                    ret.append([n] + x)
        return ret


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

    @m.Param(
        m.SIMPLICES, m.VERTEX_INDICES, m.SIMPLICES, m.VERTEX_INDICES, domain=Binary
    )
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
        return sum(
            m.vertex_is_last[i, n]
            * m.vertex_is_first[j, k]
            * m.TestVerticesEqual[i, n, j, k]
            for n in m.VERTEX_INDICES
            for k in m.VERTEX_INDICES
        ) >= sum(
            m.x[i, p] * m.x[j, p + 1] for p in m.SIMPLICES if p != m.SimplicesCount - 1
        )

    # Trivial objective (do I need this?)
    m.obj = Objective(expr=0)

    # Solve model
    results = SolverFactory(subsolver).solve(m, tee=True)
    match (results.solver.termination_condition):
        case TerminationCondition.infeasible:
            raise ValueError(
                "The triangulation was impossible to suitably order for the incremental transformation. Try a different triangulation, such as J1."
            )
        case TerminationCondition.optimal:
            pass
        case _:
            raise ValueError(
                f"Failed to generate suitable ordering for incremental transformation due to unexpected solver termination condition {results.solver.termination_condition}"
            )

    # Retrieve data
    # m.pprint()
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
def get_incremental_simplex_ordering_assume_connected_by_n_face(
    simplices, connected_face_dim, subsolver='gurobi'
):
    if connected_face_dim == 0:
        return get_incremental_simplex_ordering(simplices)
    # if not nx_available:
    #    raise ImportError('Missing Networkx')
    # G = nx.Graph()
    # G.add_nodes_from(range(len(simplices)))
    # for i in range(len(simplices)):
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

    @m.Param(
        m.SIMPLICES, m.VERTEX_INDICES, m.SIMPLICES, m.VERTEX_INDICES, domain=Binary
    )
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
        neighbors = [
            s
            for s in m.SIMPLICES
            if sum(
                m.TestVerticesEqual[i, n, s, k]
                for n in m.VERTEX_INDICES
                for k in m.VERTEX_INDICES
            )
            >= connected_face_dim + 1
            and s != i
        ]
        # print(f'neighbors of {i} are {neighbors}')
        return (
            sum(
                m.x[i, j] * m.x[k, j + 1]
                for j in m.SIMPLICES
                if j != m.SimplicesCount - 1
                for k in neighbors
            )
            + m.x[i, m.SimplicesCount - 1]
            == 1
        )

    # Trivial objective (do I need this?)
    m.obj = Objective(expr=0)

    # m.pprint()
    # Solve model
    results = SolverFactory(subsolver).solve(m, tee=True)
    match (results.solver.termination_condition):
        case TerminationCondition.infeasible:
            raise ValueError(
                f"The triangulation was impossible to suitably order for the incremental transformation under the assumption that consecutive simplices share {connected_face_dim}-faces. Try relaxing that assumption, or try a different triangulation, such as J1."
            )
        case TerminationCondition.optimal:
            pass
        case _:
            raise ValueError(
                f"Failed to generate suitable ordering for incremental transformation due to unexpected solver termination condition {results.solver.termination_condition}"
            )

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


# G_n is the graph on n! vertices where the vertices are permutations in S_n and
# two vertices are adjacent if they are related by swapping the values of
# pi(i - 1) and pi(i) for some i in {2, ..., n}.
#
# This function gets a hamiltonian path through G_n, starting from a fixed
# starting permutation, such that a fixed target symbol is either the image
# rho(1), or it is rho(n), depending on whether first or last is requested,
# where rho is the final permutation.
def get_Gn_hamiltonian(n, start_permutation, target_symbol, last):
    if n < 4:
        raise ValueError("n must be at least 4 for this operation to be possible")
    # first is enough because we can just reverse every permutation
    if last:
        return [
            tuple(reversed(pi))
            for pi in get_Gn_hamiltonian(
                n, tuple(reversed(start_permutation)), target_symbol, False
            )
        ]
    # trivial start permutation is enough because we can map it through at the end
    if start_permutation != tuple(range(1, n + 1)):
        new_target_symbol = [
            x for x in range(1, n + 1) if start_permutation[x - 1] == target_symbol
        ][0]  # pi^-1(j)
        return [
            tuple(start_permutation[pi[i] - 1] for i in range(n))
            for pi in _get_Gn_hamiltonian(n, new_target_symbol)
        ]
    else:
        return _get_Gn_hamiltonian(n, target_symbol)


# Assume the starting permutation is (1, ..., n) and the target symbol needs to
# be in the first position of the last permutation
def _get_Gn_hamiltonian(n, target_symbol):
    # base case: proof by picture from Todd, Figure 2
    # note: Figure 2 contains an error, like half the figures and paragraphs do
    if n == 4:
        if target_symbol == 1:
            return [
                (1, 2, 3, 4),
                (2, 1, 3, 4),
                (2, 1, 4, 3),
                (2, 4, 1, 3),
                (4, 2, 1, 3),
                (4, 2, 3, 1),
                (2, 4, 3, 1),
                (2, 3, 4, 1),
                (2, 3, 1, 4),
                (3, 2, 1, 4),
                (3, 2, 4, 1),
                (3, 4, 2, 1),
                (4, 3, 2, 1),
                (4, 3, 1, 2),
                (3, 4, 1, 2),
                (3, 1, 4, 2),
                (3, 1, 2, 4),
                (1, 3, 2, 4),
                (1, 3, 4, 2),
                (1, 4, 3, 2),
                (4, 1, 3, 2),
                (4, 1, 2, 3),
                (1, 4, 2, 3),
                (1, 2, 4, 3),
            ]
        elif target_symbol == 2:
            return [
                (1, 2, 3, 4),
                (1, 2, 4, 3),
                (1, 4, 2, 3),
                (4, 1, 2, 3),
                (4, 1, 3, 2),
                (1, 4, 3, 2),
                (1, 3, 4, 2),
                (1, 3, 2, 4),
                (3, 1, 2, 4),
                (3, 1, 4, 2),
                (3, 4, 1, 2),
                (4, 3, 1, 2),
                (4, 3, 2, 1),
                (3, 4, 2, 1),
                (3, 2, 4, 1),
                (3, 2, 1, 4),
                (2, 3, 1, 4),
                (2, 3, 4, 1),
                (2, 4, 3, 1),
                (4, 2, 3, 1),
                (4, 2, 1, 3),
                (2, 4, 1, 3),
                (2, 1, 4, 3),
                (2, 1, 3, 4),
            ]
        elif target_symbol == 3:
            return [
                (1, 2, 3, 4),
                (1, 2, 4, 3),
                (1, 4, 2, 3),
                (4, 1, 2, 3),
                (4, 1, 3, 2),
                (1, 4, 3, 2),
                (1, 3, 4, 2),
                (1, 3, 2, 4),
                (3, 1, 2, 4),
                (3, 1, 4, 2),
                (3, 4, 1, 2),
                (4, 3, 1, 2),
                (4, 3, 2, 1),
                (3, 4, 2, 1),
                (3, 2, 4, 1),
                (2, 3, 4, 1),
                (2, 4, 3, 1),
                (4, 2, 3, 1),
                (4, 2, 1, 3),
                (2, 4, 1, 3),
                (2, 1, 4, 3),
                (2, 1, 3, 4),
                (2, 3, 1, 4),
                (3, 2, 1, 4),
            ]
        elif target_symbol == 4:
            return [
                (1, 2, 3, 4),
                (2, 1, 3, 4),
                (2, 3, 1, 4),
                (3, 2, 1, 4),
                (3, 1, 2, 4),
                (1, 3, 2, 4),
                (1, 3, 4, 2),
                (3, 1, 4, 2),
                (3, 4, 1, 2),
                (3, 4, 2, 1),
                (3, 2, 4, 1),
                (2, 3, 4, 1),
                (2, 4, 3, 1),
                (2, 4, 1, 3),
                (2, 1, 4, 3),
                (1, 2, 4, 3),
                (1, 4, 2, 3),
                (1, 4, 3, 2),
                (4, 1, 3, 2),
                (4, 3, 1, 2),
                (4, 3, 2, 1),
                (4, 2, 3, 1),
                (4, 2, 1, 3),
                (4, 1, 2, 3),
            ]
        # unreachable
    else:
        # recursive case
        if target_symbol < n: # non-awful case
            # Well, it's still pretty awful.
            idx = n - 1
            facing = -1
            ret = []
            for pi in _get_Gn_hamiltonian(n - 1, target_symbol):
                for _ in range(n):
                    l = list(pi)
                    l.insert(idx, n)
                    ret.append(tuple(l))
                    idx += facing
                    if (idx == -1 or idx == n): # went too far
                        facing *= -1
                        idx += facing # stay once because we get a new pi
            return ret
        else: # awful case, target_symbol = n
            idx = 0
            facing = 1
            ret = []
            for pi in _get_Gn_hamiltonian(n - 1, n - 1):
                for _ in range(n):
                    l = [x + 1 for x in pi]
                    l.insert(idx, 1)
                    ret.append(tuple(l))
                    idx += facing
                    if (idx == -1 or idx == n): # went too far
                        facing *= -1
                        idx += facing # stay once because we get a new pi
            # now we almost have a correct sequence, but it ends with (1, n, ...)
            # instead of (n, 1, ...) so we need to do some surgery
            last = ret.pop() # of form (1, n, i, j, ...)
            second_last = ret.pop() # of form (n, 1, i, j, ...)
            i = last[2]
            j = last[3]
            test = list(last) # want permutation of form (n, 1, j, i, ...) with same tail
            test[0] = n
            test[1] = 1
            test[2] = j
            test[3] = i
            idx = ret.index(tuple(test))
            ret.insert(idx, second_last)
            ret.insert(idx, last)
            return ret



