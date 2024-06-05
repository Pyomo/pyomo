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
from pyomo.contrib.piecewise.triangulations import fix_vertices_incremental_order


# Set up a MIP (err, MIQCP) that orders our simplices and their vertices for us
# in the following way:
#
# (1) The simplices are ordered T_1, ..., T_N such that T_i has nonempty intersection
#     with T_{i+1}. It doesn't have to be a whole face; just a vertex is enough.
# (2) On each simplex T_i, the vertices are ordered T_i^1, ..., T_i^n such
#     that T_i^n = T_{i+1}^1
#
# Note that (2) implies (1), so we only need to enforce that.
def reorder_simplices_for_incremental(simplices, subsolver='gurobi'):
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


# An alternative approach is to order the simplices instead of the vertices. To
# do this, the condition (1) should be that they share a 1-face, not just a
# vertex. Then there is always a consistent way to choose distinct first and
# last vertices, which would otherwise be the issue - the rest of the vertex
# ordering can be arbitrary. By assuming we share 1- or n-faces, the problem
# is made somewhat smaller.
# Note that this case is literally just asking Gurobi to get a hamiltonian path
# in a large graph and hoping it can do it.
def reorder_simplices_for_incremental_assume_connected_by_n_face(
    simplices, connected_face_dim, subsolver='gurobi'
):
    if connected_face_dim == 0:
        return reorder_simplices_for_incremental(simplices)

    m = ConcreteModel()

    # Sets and Params
    m.SimplicesCount = Param(initialize=len(simplices))
    m.SIMPLICES = RangeSet(0, m.SimplicesCount - 1)
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
