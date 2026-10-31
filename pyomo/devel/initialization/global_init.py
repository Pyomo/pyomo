# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.core.base.block import BlockData
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.devel.initialization.utils import get_solution_limit_options

import logging

logger = logging.getLogger(__name__)


def _initialize_with_global_solver(
    nlp: BlockData, global_solver: SolverBase, nlp_solver: SolverBase
):
    # Check if time limit is provided for global solver
    if global_solver.config.time_limit is None:
        logger.warning(
            'No time limit set for global optimizer. '
            'For a large model, this may take a long time. '
            'Consider setting a time limit using global_solver.config.time_limit.'
        )

    # Set solution limit
    solver_options = get_solution_limit_options(global_solver)

    res = global_solver.solve(
        nlp,
        load_solutions=False,
        raise_exception_on_nonoptimal_result=False,
        solver_options=solver_options,
    )
    logger.info(
        f'solved NLP with {global_solver.name}: {res.solution_status}, {res.termination_condition}'
    )
    if res.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
        res.solution_loader.load_vars()

    return res
