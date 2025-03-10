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

import logging

from pyomo.core.base.constraint import Constraint
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import TemporarySubsystemManager, generate_subsystem_blocks
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    _generate_variables_in_constraints,
)
from pyomo.contrib.incidence_analysis.config import IncidenceMethod


_log = logging.getLogger(__name__)


def generate_strongly_connected_components(
    constraints, variables=None, include_fixed=False, igraph=None
):
    """Yield in order ``BlockData`` that each contain the variables and
    constraints of a single diagonal block in a block lower triangularization
    of the incidence matrix of constraints and variables

    These diagonal blocks correspond to strongly connected components of the
    bipartite incidence graph, projected with respect to a perfect matching
    into a directed graph.

    Parameters
    ----------
    constraints: List of Pyomo constraint data objects
        Constraints used to generate strongly connected components.
    variables: List of Pyomo variable data objects
        Variables that may participate in strongly connected components.
        If not provided, all variables in the constraints will be used.
    include_fixed: Bool, optional
        Indicates whether fixed variables will be included when
        identifying variables in constraints.
    igraph: IncidenceGraphInterface, optional
        Incidence graph containing (at least) the provided constraints
        and variables.

    Yields
    ------
    Tuple of ``BlockData``, list-of-variables
        Blocks containing the variables and constraints of every strongly
        connected component, in a topological order. The variables are the
        "input variables" for that block.

    """
    if variables is None:
        variables = list(
            _generate_variables_in_constraints(
                constraints,
                include_fixed=include_fixed,
                method=IncidenceMethod.ampl_repn,
            )
        )

    if len(variables) != len(constraints):
        nvar = len(variables)
        ncon = len(constraints)
        raise RuntimeError(
            "generate_strongly_connected_components only supports systems with the"
            f" same numbers of variables and equality constraints. Got {nvar}"
            f" variables and {ncon} constraints."
        )
    if igraph is None:
        igraph = IncidenceGraphInterface()

    var_blocks, con_blocks = igraph.block_triangularize(
        variables=variables, constraints=constraints
    )
    subsets = [(cblock, vblock) for vblock, cblock in zip(var_blocks, con_blocks)]
    for block, inputs in generate_subsystem_blocks(
        subsets, include_fixed=include_fixed
    ):
        # TODO: How does len scale for reference-to-list?
        # If this assert fails, it may be due to a bug in block_triangularize
        # or generate_subsystem_block.
        assert len(block.vars) == len(block.cons)
        yield (block, inputs)


def solve_strongly_connected_components(
    block, *, solver=None, solve_kwds=None, use_calc_var=True, calc_var_kwds=None
):
    """Solve a square system of variables and equality constraints by
    solving strongly connected components individually.

    Strongly connected components (of the directed graph of constraints
    obtained from a perfect matching of variables and constraints) are
    the diagonal blocks in a block triangularization of the incidence
    matrix, so solving the strongly connected components in topological
    order is sufficient to solve the entire block.

    One-by-one blocks are solved using Pyomo's
    calculate_variable_from_constraint function, while higher-dimension
    blocks are solved using the user-provided solver object.

    Parameters
    ----------
    block: Pyomo Block
        The Pyomo block whose variables and constraints will be solved
    solver: Pyomo solver object
        The solver object that will be used to solve strongly connected
        components of size greater than one constraint. Must implement
        a solve method.
    solve_kwds: Dictionary
        Keyword arguments for the solver's solve method
    use_calc_var: Bool
        Whether to use ``calculate_variable_from_constraint`` for one-by-one
        square system solves
    calc_var_kwds: Dictionary
        Keyword arguments for calculate_variable_from_constraint

    Returns
    -------
    List of results objects returned by each call to solve

    """
    if solve_kwds is None:
        solve_kwds = {}
    if calc_var_kwds is None:
        calc_var_kwds = {}

    igraph = IncidenceGraphInterface(
        block,
        active=True,
        include_fixed=False,
        include_inequality=False,
        method=IncidenceMethod.ampl_repn,
    )
    constraints = igraph.constraints
    variables = igraph.variables

    res_list = []
    log_blocks = _log.isEnabledFor(logging.DEBUG)
    for scc, inputs in generate_strongly_connected_components(
        constraints, variables, igraph=igraph
    ):
        with TemporarySubsystemManager(to_fix=inputs, remove_bounds_on_fix=True):
            N = len(scc.vars)
            if N == 1 and use_calc_var:
                if log_blocks:
                    _log.debug(f"Solving 1x1 block: {scc.cons[0].name}.")
                results = calculate_variable_from_constraint(
                    scc.vars[0], scc.cons[0], **calc_var_kwds
                )
            else:
                if solver is None:
                    var_names = [var.name for var in scc.vars.values()][:10]
                    con_names = [con.name for con in scc.cons.values()][:10]
                    raise RuntimeError(
                        "An external solver is required if block has strongly\n"
                        "connected components of size greater than one (is not"
                        " a DAG).\nGot an SCC of size %sx%s including"
                        " components:\n%s\n%s" % (N, N, var_names, con_names)
                    )
                if log_blocks:
                    _log.debug(f"Solving {N}x{N} block.")
                results = solver.solve(scc, **solve_kwds)
            res_list.append(results)
    return res_list
