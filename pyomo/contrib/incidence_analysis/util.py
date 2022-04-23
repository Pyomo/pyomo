#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
        create_subsystem_block,
        TemporarySubsystemManager,
        generate_subsystem_blocks,
        )
from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface


def generate_strongly_connected_components(
        constraints,
        variables=None,
        include_fixed=False,
        ):
    """ Performs a block triangularization of the incidence matrix
    of the provided constraints and variables, and yields a block that
    contains the constraints and variables of each diagonal block
    (strongly connected component).

    Arguments
    ---------
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
    Blocks containing the variables and constraints of every strongly
    connected component, in a topological order, as well as the
    "input variables" for that block

    """
    if variables is None:
        var_set = ComponentSet()
        variables = []
        for con in constraints:
            for var in identify_variables(
                    con.expr,
                    include_fixed=include_fixed,
                    ):
                if var not in var_set:
                    variables.append(var)
                    var_set.add(var)

    assert len(variables) == len(constraints)
    igraph = IncidenceGraphInterface()
    var_block_map, con_block_map = igraph.block_triangularize(
            variables=variables,
            constraints=constraints,
            )
    blocks = set(var_block_map.values())
    n_blocks = len(blocks)
    var_blocks = [[] for b in range(n_blocks)]
    con_blocks = [[] for b in range(n_blocks)]
    for var, b in var_block_map.items():
        var_blocks[b].append(var)
    for con, b in con_block_map.items():
        con_blocks[b].append(con)
    subsets = list(zip(con_blocks, var_blocks))
    for block, inputs in generate_subsystem_blocks(
            subsets,
            include_fixed=include_fixed,
            ):
        # TODO: How does len scale for reference-to-list?
        assert len(block.vars) == len(block.cons)
        yield (block, inputs)


def solve_strongly_connected_components(
        block,
        solver=None,
        solve_kwds=None,
        calc_var_kwds=None,
        ):
    """ This function solves a square block of variables and equality
    constraints by solving strongly connected components individually.
    Strongly connected components (of the directed graph of constraints
    obtained from a perfect matching of variables and constraints) are
    the diagonal blocks in a block triangularization of the incidence
    matrix, so solving the strongly connected components in topological
    order is sufficient to solve the entire block.

    One-by-one blocks are solved using Pyomo's
    calculate_variable_from_constraint function, while higher-dimension
    blocks are solved using the user-provided solver object.

    Arguments
    ---------
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

    constraints = list(block.component_data_objects(Constraint, active=True))
    var_set = ComponentSet()
    variables = []
    for con in constraints:
        for var in identify_variables(con.expr, include_fixed=False):
            # Because we are solving, we do not want to include fixed variables
            if var not in var_set:
                variables.append(var)
                var_set.add(var)

    res_list = []
    for scc, inputs in generate_strongly_connected_components(
            constraints,
            variables,
            ):
        with TemporarySubsystemManager(to_fix=inputs):
            if len(scc.vars) == 1:
                results = calculate_variable_from_constraint(
                    scc.vars[0], scc.cons[0], **calc_var_kwds
                )
                res_list.append(results)
            else:
                if solver is None:
                    # NOTE: Use local name to avoid slow generation of this
                    # error message if a user provides a large, non-decomposable
                    # block with no solver.
                    vars = [var.local_name for var in scc.vars.values()]
                    cons = [con.local_name for con in scc.cons.values()]
                    raise RuntimeError(
                        "An external solver is required if block has strongly\n"
                        "connected components of size greater than one (is not "
                        "a DAG).\nGot an SCC with components: \n%s\n%s"
                        % (vars, cons)
                        )
                results = solver.solve(scc, **solve_kwds)
                res_list.append(results)
    return res_list
