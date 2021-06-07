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
from pyomo.core.base.constraint import constraint
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
        create_subsystem_block,
        TemporarySubsystemManager,
        )
from pyomo.contrib.incidence_analysis.interface import IncidenceGraphInterface


def solve_strongly_connected_components(block, solver=None, solver_kwds=None):
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
    solver_kwds: Keyword arguments for the solver's solve method

    """
    if solver_kwds is None:
        solver_kwds = {}

    variables = [var for var in block.component_data_objects(Var)
            if not var.fixed]
    constraints = [con for con in block.component_data_objects(Constraint)
            if con.active]
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
    for b_vars, b_cons in zip(var_blocks, con_blocks):
        assert len(b_vars) == len(b_cons)
        if len(b_vars) == 1:
            var = b_vars[0]
            con = b_cons[0]
            calculate_variable_from_constraint(var, con)
        else:
            _temp = create_subsystem_block(b_vars, b_cons)
            with TemporarySubsystemManager(to_fix=list(_temp.other_vars[:])):
                solver.solve(_temp, **solver_kwds)
