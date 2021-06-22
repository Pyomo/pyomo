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
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
        create_subsystem_block,
        TemporarySubsystemManager,
        generate_subsystem_blocks,
        )
from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface


def generate_strongly_connected_components(
        block,
        include_fixed=False,
        ):
    """ Performs a block triangularization of the variable-constraint
    incidence matrix of the provided block, and yields a block that
    contains the variables and constraints of each diagonal block
    (strongly connected component).

    Arguments
    ---------
    block: Block whose strongly connected components will be generated
    include_fixed: Bool indicating whether fixed variables will be
                   attached as "input variables" on the subsystem blocks
                   containing the strongly connected components
    fix_inputs: Bool indicating whether "input variables" on a subsystem
                block should be temporarily fixed when the block is
                yielded. The default is True as this is necessary to solve
                the strongly connected component as a square system.

    Yields
    ------
    Blocks containing the variables and constraints of every strongly
    connected component, in a topological order

    """
    variables = [var for var in block.component_data_objects(Var)
            if not var.fixed]
    constraints = [con for con in 
            block.component_data_objects(Constraint, active=True)]
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
    for block in generate_subsystem_blocks(
            subsets,
            include_fixed=include_fixed,
            ):
        # TODO: How does len scale for reference-to-list?
        assert len(block.vars) == len(block.cons)
        yield block


def solve_strongly_connected_components(block, solver=None, solve_kwds=None):
    """ This function solves a square block of variables and equality
    constraints by solving strongly connected components individually.
    Strongly connected components (of the directed graph of constraints
    obtained from a perfect matching of variables and constraints) are
    the diagonal blocks in a block triangularization of the incidence
    matrix, so solving the strongly connected components in topological
    order is sufficient to solve the entire block.

    Arguments
    ---------
    block: The Pyomo block whose variables and constraints will be solved
    solver: The solver object that will be used to solve strongly connected
            components of size greater than one constraint. Must implement
            a solve method.
    solve_kwds: Keyword arguments for the solver's solve method

    """
    if solve_kwds is None:
        solve_kwds = {}

    res_list = []
    for scc in generate_strongly_connected_components(block):
        inputs = list(scc.input_vars.values())
        with TemporarySubsystemManager(to_fix=inputs):
            if len(scc.vars) == 1:
                calculate_variable_from_constraint(scc.vars[0], scc.cons[0])
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
