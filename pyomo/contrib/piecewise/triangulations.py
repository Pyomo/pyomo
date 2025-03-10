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

import itertools
from enum import Enum
from pyomo.common.errors import DeveloperError
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.piecewise.ordered_3d_j1_triangulation_data import (
    get_hamiltonian_paths,
)


class Triangulation(Enum):
    Unknown = 0
    AssumeValid = 1
    Delaunay = 2
    J1 = 3
    OrderedJ1 = 4


# Duck-typed thing that looks reasonably similar to an instance of
# scipy.spatial.Delaunay
# Fields:
#   - points: list of P points as P x n array
#   - simplices: list of M simplices as P x (n + 1) array of point _indices_
#   - coplanar: list of N points omitted from triangulation as tuples of (point index,
#     nearest simplex index, nearest vertex index), stacked into an N x 3 array
class _Triangulation:
    def __init__(self, points, simplices, coplanar):
        self.points = points
        self.simplices = simplices
        self.coplanar = coplanar


# Get an unordered J1 triangulation, as described by [1], of a finite grid of
# points in R^n having the same odd number of points along each axis.
# References
# ----------
# [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
#     for nonseparable piecewise-linear optimization: unifying framework
#     and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
#     2010.
def get_unordered_j1_triangulation(points, dimension):
    points_map, num_pts = _process_points_j1(points, dimension)
    simplices_list = _get_j1_triangulation(points_map, num_pts - 1, dimension)
    return _Triangulation(
        points=np.array(points),
        simplices=np.array(simplices_list),
        coplanar=np.array([]),
    )


# Get an ordered J1 triangulation, according to [1], with the additional condition
# added from [2] that simplex vertices are also ordered such that the final vertex
# of each simplex is the first vertex of the next simplex.
# References
# ----------
# [1] Michael J. Todd. "Hamiltonian triangulations of Rn". In: Functional
#     Differential Equations and Approximation of Fixed Points. Ed. by
#     Heinz-Otto Peitgen and Hans-Otto Walther. Berlin, Heidelberg: Springer
#     Berlin Heidelberg, 1979, pp. 470–483. ISBN: 978-3-540-35129-0.
# [2] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
#     for nonseparable piecewise-linear optimization: unifying framework
#     and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
#     2010.
def get_ordered_j1_triangulation(points, dimension):
    points_map, num_pts = _process_points_j1(points, dimension)
    if dimension == 2:
        simplices_list = _get_ordered_j1_triangulation_2d(points_map, num_pts - 1)
    elif dimension == 3:
        simplices_list = _get_ordered_j1_triangulation_3d(points_map, num_pts - 1)
    else:
        simplices_list = _get_ordered_j1_triangulation_4d_and_above(
            points_map, num_pts - 1, dimension
        )
    return _Triangulation(
        points=np.array(points),
        simplices=np.array(simplices_list),
        coplanar=np.array([]),
    )


# Does some validation but mostly assumes the user did the right thing
def _process_points_j1(points, dimension):
    if not len(points[0]) == dimension:
        raise ValueError("Points not consistent with specified dimension")
    num_pts = round(len(points) ** (1 / dimension))
    if not len(points) == num_pts**dimension:
        raise ValueError(
            "'points' must have points forming an n-dimensional grid with straight grid"
            " lines and the same odd number of points in each axis."
        )
    if not num_pts % 2 == 1:
        raise ValueError(
            "'points' must have points forming an n-dimensional grid with straight grid"
            " lines and the same odd number of points in each axis."
        )

    # munge the points into an organized map from n-dimensional keys to original
    # indices
    points.sort()
    points_map = {}
    for point_index in itertools.product(range(num_pts), repeat=dimension):
        point_flat_index = 0
        for n in range(dimension):
            point_flat_index += point_index[dimension - 1 - n] * num_pts**n
        points_map[point_index] = point_flat_index
    return points_map, num_pts


# Implement the J1 "Union Jack" triangulation (Todd 79) as explained by
# Vielma 2010, with no ordering guarantees imposed. This function triangulates
# {0, ..., K}^n for even K using the J1 triangulation, mapping the
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
        simplex.append(points_map[tuple(current)])
        for i in range(0, n):
            current[pi[i]] += s[pi[i]]
            simplex.append(points_map[tuple(current)])
        # sort this because it might happen again later and we'd like to stay
        # consistent. Undo this if it's slow.
        ret.append(sorted(simplex))
    return ret


class Direction(Enum):
    left = 0
    down = 1
    up = 2
    right = 3


# Implement something similar to proof-by-picture from Todd 79 (Figure 1).
# However, that drawing is misleading at best so I do it in a working way, and
# also slightly more regularly. I also go from the outside in instead of from
# the inside out, to make things easier to implement.
def _get_ordered_j1_triangulation_2d(points_map, num_pts):
    # check when square has simplices in top-left and bottom-right
    square_parity_tlbr = lambda x, y: x % 2 == y % 2
    # check when we are in a "turnaround square" as seen in the picture
    is_turnaround = lambda x, y: x >= num_pts / 2 and y == (num_pts / 2) - 1

    facing = None

    simplices = []
    start_square = (num_pts - 1, (num_pts / 2) - 1)

    # make it easier to read what I'm doing
    def add_bottom_right():
        simplices.append(
            (points_map[x, y], points_map[x + 1, y], points_map[x + 1, y + 1])
        )

    def add_top_right():
        simplices.append(
            (points_map[x, y + 1], points_map[x + 1, y], points_map[x + 1, y + 1])
        )

    def add_bottom_left():
        simplices.append((points_map[x, y], points_map[x, y + 1], points_map[x + 1, y]))

    def add_top_left():
        simplices.append(
            (points_map[x, y], points_map[x, y + 1], points_map[x + 1, y + 1])
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
        if facing == Direction.left:
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
        elif facing == Direction.right:
            if is_turnaround(x, y):
                # finished; this case should always eventually be reached
                add_bottom_left()
                _fix_vertices_incremental_order(simplices)
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
        elif facing == Direction.down:
            if is_turnaround(x, y):
                # we are always in a TLBR square. Take the TL of this, the TR
                # of the one on the left, and continue upwards one to the left
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
        elif facing == Direction.up:
            if is_turnaround(x, y):
                # we are always in a non-TLBR square. Take the BL of this, the BR
                # of the one on the left, and continue downwards one to the left
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
    incremental_3d_simplex_pair_to_path = get_hamiltonian_paths()
    # To start, we need a hamiltonian path in the grid graph of *double* cubes
    # (2x2x2 cubes)
    grid_hamiltonian = _get_grid_hamiltonian(3, round(num_pts / 2))  # division is exact

    # We always start by going from [0, 0, 0] to [0, 0, 1], so we can safely
    # start from the -x side.
    # Data format: the first tuple is a basis vector or its negative, representing a
    # face. The number afterwards is a 1 or 2 disambiguating which, of the two simplices
    # on that face we consider, we are referring to.
    start_data = ((-1, 0, 0), 1)

    simplices = []
    for i in range(len(grid_hamiltonian) - 1):
        current_double_cube_idx = grid_hamiltonian[i]
        next_double_cube_idx = grid_hamiltonian[i + 1]
        direction_to_next = tuple(
            next_double_cube_idx[j] - current_double_cube_idx[j] for j in range(3)
        )

        current_v_0 = tuple(2 * current_double_cube_idx[j] + 1 for j in range(3))

        current_cube_path = None
        if (
            start_data,
            (direction_to_next, 1),
        ) in incremental_3d_simplex_pair_to_path.keys():
            current_cube_path = incremental_3d_simplex_pair_to_path[
                (start_data, (direction_to_next, 1))
            ]
            # set the start data for the next iteration now
            start_data = (tuple(-1 * i for i in direction_to_next), 1)
        else:
            current_cube_path = incremental_3d_simplex_pair_to_path[
                (start_data, (direction_to_next, 2))
            ]
            start_data = (tuple(-1 * i for i in direction_to_next), 2)

        for simplex_data in current_cube_path:
            simplices.append(
                _get_one_j1_simplex(
                    current_v_0, simplex_data[1], simplex_data[0], 3, points_map
                )
            )

    # fill in the last cube. We have a good start_data but we need to invent a
    # direction_to_next. Let's go straight in the direction we came from.
    direction_to_next = tuple(-1 * i for i in start_data[0])
    current_v_0 = tuple(2 * grid_hamiltonian[-1][j] + 1 for j in range(3))
    if (
        start_data,
        (direction_to_next, 1),
    ) in incremental_3d_simplex_pair_to_path.keys():
        current_cube_path = incremental_3d_simplex_pair_to_path[
            (start_data, (direction_to_next, 1))
        ]
    else:
        current_cube_path = incremental_3d_simplex_pair_to_path[
            (start_data, (direction_to_next, 2))
        ]

    for simplex_data in current_cube_path:
        simplices.append(
            _get_one_j1_simplex(
                current_v_0, simplex_data[1], simplex_data[0], 3, points_map
            )
        )

    _fix_vertices_incremental_order(simplices)
    return simplices


def _get_ordered_j1_triangulation_4d_and_above(points_map, num_pts, dim):
    # step one: get a hamiltonian path in the appropriate grid graph (low-coordinate
    # corners of the grid squares)
    grid_hamiltonian = _get_grid_hamiltonian(dim, num_pts)

    # step 1.5: get a starting simplex. Anything that is *not* adjacent to the
    # second square is fine. Since we always go from [0, ..., 0] to [0, ..., 1],
    # i.e., j=`dim`, anything where `dim` is not the first or last symbol should
    # always work. Let's stick it in the second place
    start_perm = tuple([1] + [dim] + list(range(2, dim)))

    # step two: for each square, get a sequence of simplices from a starting simplex,
    # through the square, and then ending with a simplex adjacent to the next square.
    # Then find the appropriate adjacent simplex to start on the next square
    simplices = []
    for i in range(len(grid_hamiltonian) - 1):
        current_corner = grid_hamiltonian[i]
        next_corner = grid_hamiltonian[i + 1]
        # differing index
        j = [k + 1 for k in range(dim) if current_corner[k] != next_corner[k]][0]
        # border x_j value between this square and next
        c = max(current_corner[j - 1], next_corner[j - 1])
        v_0, sign = _get_nearest_odd_and_sign_vec(current_corner)
        # According to Todd, what we need is to end with a permutation where rho(n) = j
        # if c is odd, and end with one where rho(1) = j if c is even. I think this
        # is right -- basically the sign from the sign vector sometimes cancels
        # out the sign from whether we are entering in the +c or -c direction.
        if c % 2 == 0:
            perm_sequence = _get_Gn_hamiltonian(dim, start_perm, j, False)
            for pi in perm_sequence:
                simplices.append(_get_one_j1_simplex(v_0, pi, sign, dim, points_map))
        else:
            perm_sequence = _get_Gn_hamiltonian(dim, start_perm, j, True)
            for pi in perm_sequence:
                simplices.append(_get_one_j1_simplex(v_0, pi, sign, dim, points_map))
        # should be true regardless of odd or even
        start_perm = perm_sequence[-1]

    # step three: finish out the last square
    # Any final permutation is fine; we are going nowhere after this
    v_0, sign = _get_nearest_odd_and_sign_vec(grid_hamiltonian[-1])
    for pi in _get_Gn_hamiltonian(dim, start_perm, 1, False):
        simplices.append(_get_one_j1_simplex(v_0, pi, sign, dim, points_map))

    # fix vertices and return
    _fix_vertices_incremental_order(simplices)
    return simplices


def _get_one_j1_simplex(v_0, pi, sign, dim, points_map):
    simplex = []
    current = list(v_0)
    simplex.append(points_map[tuple(current)])
    for i in range(0, dim):
        current[pi[i] - 1] += sign[pi[i] - 1]
        simplex.append(points_map[tuple(current)])
    return sorted(simplex)


# get the v_0 and sign vectors corresponding to a given square, identified by its
# low-coordinate corner
def _get_nearest_odd_and_sign_vec(corner):
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


def _get_grid_hamiltonian(dim, length):
    if dim == 1:
        return [[n] for n in range(length)]
    else:
        ret = []
        prev = _get_grid_hamiltonian(dim - 1, length)
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


# Fix vertices (in place) when the simplices are right but vertices are not
def _fix_vertices_incremental_order(simplices):
    last_vertex_index = len(simplices[0]) - 1
    for i, simplex in enumerate(simplices):
        # Choose vertices like this: first is always the same as last
        # of the previous simplex. Last is arbitrarily chosen from the
        # intersection with the next simplex.
        first = None
        last = None
        if i == 0:
            first = 0
        else:
            first = simplex.index(simplices[i - 1][last_vertex_index])

        if i == len(simplices) - 1:
            last = last_vertex_index
        else:
            for n in range(last_vertex_index + 1):
                if simplex[n] in simplices[i + 1] and n != first:
                    last = n
                    break
            else:
                # For the Python neophytes in the audience (and other sane
                # people), the 'else' only runs if we do *not* break out of the
                # for loop.
                raise DeveloperError("Couldn't fix vertex ordering for incremental.")

        # reorder the simplex with the desired first and last
        new_simplex = list(simplex)
        temp = new_simplex[0]
        new_simplex[0] = new_simplex[first]
        new_simplex[first] = temp
        if last == 0:
            last = first
        temp = new_simplex[last_vertex_index]
        new_simplex[last_vertex_index] = new_simplex[last]
        new_simplex[last] = temp
        simplices[i] = tuple(new_simplex)


# Let G_n be the graph on n! vertices where the vertices are permutations in
# S_n and  two vertices are adjacent if they are related by swapping the values
# of pi(i - 1) and pi(i) for some i in {2, ..., n}.
#
# This function gets a Hamiltonian path through G_n, starting from a fixed
# starting permutation, such that a fixed target symbol is either the image
# rho(1), or it is rho(n), depending on whether first or last is requested,
# where rho is the final permutation.
def _get_Gn_hamiltonian(n, start_permutation, target_symbol, last, _cache={}):
    if n < 4:
        raise ValueError("n must be at least 4 for this operation to be possible")
    if (n, start_permutation, target_symbol, last) in _cache:
        return _cache[(n, start_permutation, target_symbol, last)]
    # first is enough because we can just reverse every permutation
    if last:
        ret = [
            tuple(reversed(pi))
            for pi in _get_Gn_hamiltonian(
                n, tuple(reversed(start_permutation)), target_symbol, False
            )
        ]
        _cache[(n, start_permutation, target_symbol, last)] = ret
        return ret
    # trivial start permutation is enough because we can map it through at the end
    if start_permutation != tuple(range(1, n + 1)):
        new_target_symbol = [
            x for x in range(1, n + 1) if start_permutation[x - 1] == target_symbol
        ][
            0
        ]  # pi^-1(j)
        ret = [
            tuple(start_permutation[pi[i] - 1] for i in range(n))
            for pi in _get_Gn_hamiltonian_impl(n, new_target_symbol)
        ]
        _cache[(n, start_permutation, target_symbol, last)] = ret
        return ret
    else:
        ret = _get_Gn_hamiltonian_impl(n, target_symbol)
        _cache[(n, start_permutation, target_symbol, last)] = ret
        return ret


# Assume the starting permutation is (1, ..., n) and the target symbol needs to
# be in the first position of the last permutation
def _get_Gn_hamiltonian_impl(n, target_symbol):
    # base case: proof by picture from Todd 79, Figure 2
    # note: Figure 2 contains an error, careful!
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
        if target_symbol < n:  # Less awful case
            idx = n - 1
            facing = -1
            ret = []
            for pi in _get_Gn_hamiltonian_impl(n - 1, target_symbol):
                for _ in range(n):
                    l = list(pi)
                    l.insert(idx, n)
                    ret.append(tuple(l))
                    idx += facing
                    if idx == -1 or idx == n:  # went too far
                        facing *= -1
                        idx += facing  # stay once because we get a new pi
            return ret
        else:  # awful case, target_symbol = n
            idx = 0
            facing = 1
            ret = []
            for pi in _get_Gn_hamiltonian_impl(n - 1, n - 1):
                for _ in range(n):
                    l = [x + 1 for x in pi]
                    l.insert(idx, 1)
                    ret.append(tuple(l))
                    idx += facing
                    if idx == -1 or idx == n:  # went too far
                        facing *= -1
                        idx += facing  # stay once because we get a new pi
            # now we almost have a correct sequence, but it ends with (1, n, ...)
            # instead of (n, 1, ...) so we need to do some surgery
            last = ret.pop()  # of form (1, n, i, j, ...)
            second_last = ret.pop()  # of form (n, 1, i, j, ...)
            i = last[2]
            j = last[3]
            test = list(
                last
            )  # want permutation of form (n, 1, j, i, ...) with same tail
            test[0] = n
            test[1] = 1
            test[2] = j
            test[3] = i
            idx = ret.index(tuple(test))
            ret.insert(idx, second_last)
            ret.insert(idx, last)
            return ret
