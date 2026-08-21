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
from pyomo.common.dependencies import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.devel.initialization.bounds.bound_variables import (
    bound_all_nonlinear_variables,
)
from pyomo.devel.initialization.utils import shallow_clone, fix_vars_with_equal_bounds
import logging

logger = logging.getLogger(__name__)


def _initialize_with_multistart_solver(
    nlp: BlockData,
    multistart_solver,
    seed,
    default_bound=1.0e8,
    use_univariate_nonlinear_decomposition: bool = True,
    aggressive_substitution: bool = True,
):

    # Make a shallow clone
    nlp = shallow_clone(nlp)

    # first introduce auxiliary variables so that we don't try to
    # approximate any functions of more than two variables
    # this does not matter as much as it does for PWL
    if use_univariate_nonlinear_decomposition:
        trans = pyo.TransformationFactory(
            'contrib.piecewise.univariate_nonlinear_decomposition'
        )
        trans.apply_to(nlp, aggressive_substitution=aggressive_substitution)
        logger.info('applied the univariate_nonlinear_decomposition transformation')

    # bounds on the nonlinear variables
    bound_all_nonlinear_variables(nlp, default_bound=default_bound)
    logger.info('bounded nonlinear variables')

    # fix variables with equal bounds for sampler
    fix_vars_with_equal_bounds(nlp)
    logger.info('fixed variables with equal bounds')

    # Set multistart solver configs
    multistart_solver.config.seed = seed
    multistart_solver.config.load_solutions = False
    multistart_solver.config.raise_exception_on_nonoptimal_result = False

    res = multistart_solver.solve(nlp)
    logger.info('Finished multistart optimization iterations.')
    if res.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
        res.solution_loader.load_vars()

    return res
