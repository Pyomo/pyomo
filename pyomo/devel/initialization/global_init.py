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
from pyomo.contrib.solver.solvers.scip.scip_direct import ScipDirect, ScipPersistent
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_minlp import GurobiDirectMINLP
import logging

logger = logging.getLogger(__name__)


def _initialize_with_global_solver(
    nlp: BlockData, global_solver: SolverBase, nlp_solver: SolverBase
):
    if isinstance(global_solver, (ScipDirect, ScipPersistent)):
        opts = {'limits/solutions': 1}
    elif isinstance(global_solver, (GurobiDirectMINLP,)):
        opts = {'SolutionLimit': 1}
    else:
        raise NotImplementedError(
            'Currently, the initialization module only works with new solver interface, so the global solvers are limited to ScipDirect, ScipPersistent, and GurobiDirectMINLP.'
        )
    res = global_solver.solve(
        nlp,
        load_solutions=True,
        raise_exception_on_nonoptimal_result=False,
        solver_options=opts,
    )
    logger.info(
        f'solved NLP with {global_solver.name}: {res.solution_status}, {res.termination_condition}'
    )
    res = nlp_solver.solve(
        nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False
    )
    logger.info(
        f'solved NLP with {nlp_solver.name}: {res.solution_status}, {res.termination_condition}'
    )
    if res.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
        res.solution_loader.load_vars()
    else:
        logger.warning('initialization was not successful via global optimization')

    return res
