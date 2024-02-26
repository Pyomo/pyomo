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

import logging

from pyomo.core.base.constraint import Constraint
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
    TemporarySubsystemManager,
    generate_subsystem_blocks,
    create_subsystem_block,
)
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    _generate_variables_in_constraints,
)
from pyomo.contrib.incidence_analysis.config import IncidenceMethod


_log = logging.getLogger(__name__)


from pyomo.common.timing import HierarchicalTimer
def generate_strongly_connected_components(
    constraints,
    variables=None,
    include_fixed=False,
    timer=None,
):
    """Yield in order ``_BlockData`` that each contain the variables and
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
    include_fixed: Bool
        Indicates whether fixed variables will be included when
        identifying variables in constraints.

    Yields
    ------
    Tuple of ``_BlockData``, list-of-variables
        Blocks containing the variables and constraints of every strongly
        connected component, in a topological order. The variables are the
        "input variables" for that block.

    """
    if timer is None:
        timer = HierarchicalTimer()

    if isinstance(constraints, IncidenceGraphInterface):
        igraph = constraints
        variables = igraph.variables
        constraints = igraph.constraints
    else:
        if variables is None:
            timer.start("generate-variables")
            variables = list(
                _generate_variables_in_constraints(constraints, include_fixed=include_fixed)
            )
            timer.stop("generate-variables")
        timer.start("igraph")
        igraph = IncidenceGraphInterface()
        timer.stop("igraph")

    assert len(variables) == len(constraints)

    timer.start("block-triang")
    var_blocks, con_blocks = igraph.block_triangularize(
        variables=variables, constraints=constraints
    )
    timer.stop("block-triang")
    subsets = [(cblock, vblock) for vblock, cblock in zip(var_blocks, con_blocks)]
    timer.start("generate-block")
    for block, inputs in generate_subsystem_blocks(
        subsets, include_fixed=include_fixed
    ):
        timer.stop("generate-block")
        # TODO: How does len scale for reference-to-list?
        assert len(block.vars) == len(block.cons)
        yield (block, inputs)
        # Note that this code, after the last yield, I believe is only called
        # at time of GC.
        timer.start("generate-block")
    timer.stop("generate-block")


def solve_strongly_connected_components(
    block,
    solver=None,
    solve_kwds=None,
    calc_var_kwds=None,
    timer=None,
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
    if timer is None:
        timer = HierarchicalTimer()

    timer.start("igraph")
    igraph = IncidenceGraphInterface(
        block,
        active=True,
        include_fixed=False,
        include_inequality=False,
        method=IncidenceMethod.ampl_repn,
    )
    timer.stop("igraph")
    # Use IncidenceGraphInterface to get the constraints and variables
    constraints = igraph.constraints
    variables = igraph.variables

    timer.start("block-triang")
    var_blocks, con_blocks = igraph.block_triangularize()
    timer.stop("block-triang")
    timer.start("subsystem-blocks")
    subsystem_blocks = [
        create_subsystem_block(conbl, varbl, timer=timer) if len(varbl) > 1 else None
        for varbl, conbl in zip(var_blocks, con_blocks)
    ]
    timer.stop("subsystem-blocks")

    res_list = []
    log_blocks = _log.isEnabledFor(logging.DEBUG)

    #timer.start("generate-scc")
    #for scc, inputs in generate_strongly_connected_components(igraph, timer=timer):
    #    timer.stop("generate-scc")
    for i, scc in enumerate(subsystem_blocks):
        if scc is None:
            # Since a block is not necessary for 1x1 solve, we use the convention
            # that None indicates a 1x1 SCC.
            inputs = []
            var = var_blocks[i][0]
            con = con_blocks[i][0]
        else:
            inputs = list(scc.input_vars.values())

        with TemporarySubsystemManager(to_fix=inputs):
            N = len(var_blocks[i])
            if N == 1:
                if log_blocks:
                    _log.debug(f"Solving 1x1 block: {scc.cons[0].name}.")
                timer.start("calc-var")
                results = calculate_variable_from_constraint(
                    #scc.vars[0], scc.cons[0], **calc_var_kwds
                    var, con, **calc_var_kwds
                )
                timer.stop("calc-var")
                res_list.append(results)
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
                timer.start("solve")
                results = solver.solve(scc, **solve_kwds)
                timer.stop("solve")
                res_list.append(results)
    #    timer.start("generate-scc")
    #timer.stop("generate-scc")
    return res_list
