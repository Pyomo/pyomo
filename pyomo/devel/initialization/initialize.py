# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from typing import Optional
from pyomo.core.base.block import BlockData
from enum import Enum
from pyomo.devel.initialization.utils import get_vars
from pyomo.common.collections import ComponentMap
from pyomo.devel.initialization.pwl_init import (
    _initialize_with_piecewise_linear_approximation,
)
from pyomo.devel.initialization.lp_approx_init import _initialize_with_LP_approximation
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.devel.initialization.global_init import _initialize_with_global_solver
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import Results
import logging
from pyomo.contrib.solver.common.results import SolutionStatus

logger = logging.getLogger(__name__)


def _get_solver(sname, reason):
    opt = SolverFactory(sname)
    if opt.available():
        logger.info(f'Using {sname} for {reason} because a solver was not specified')
    else:
        raise RuntimeError(
            f'No solver was specified for {reason} and the default ({sname}) is not available'
        )
    return opt


def _setup(nlp):
    # get all variable bounds, domains, etc. to restore them later
    orig_vars = get_vars(nlp)
    orig_var_data = [
        (v, (v.lower, v.upper, v.domain, v.fixed, v.value)) for v in orig_vars
    ]
    for v, vdata in orig_var_data:
        if vdata[2].isdiscrete():
            raise RuntimeError(
                'Initialization module currently only supports continuous models.'
            )
    return orig_var_data


def _cleanup(orig_var_data):
    # restore variable bounds, domain, etc.
    for v, (lb, ub, domain, fixed, value) in orig_var_data:
        v.setlb(lb)
        v.setub(ub)
        v.domain = domain
        if fixed:
            assert v.value == value
            assert v.fixed
        else:
            v.unfix()


def _try_nlp_solve(nlp: BlockData, nlp_solver: SolverBase):
    # try to solve the nlp before doing extra work
    res = nlp_solver.solve(
        nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False
    )
    logger.info(f'solved NLP: {res.solution_status}, {res.termination_condition}')

    if res.solution_status == SolutionStatus.optimal:
        res.solution_loader.load_vars()
        logger.info('NLP solved without any initialization')
    return res


def initialize_with_piecewise_linear_approximation(
    nlp: BlockData,
    nlp_solver: SolverBase | None = None,
    mip_solver: SolverBase | None = None,
    default_bound: float = 1.0e8,
    max_pwl_refinement_iter: int = 100,
    num_pwl_cons_to_refine_per_iter: int = 5,
    aggressive_substitution: bool = True,
    skip_initial_nlp_solve: bool = False,
    bounds_tol: float = 1e-6,
) -> Results:
    """
    Attempt to initialize the problem with a piecewise linear approximation
    and subsequently solve the model given by ``nlp``.

    Parameters
    ----------
    nlp: BlockData
        The pyomo model to be initialized.
    nlp_solver: Optional[SolverBase]
        A solver interface appropriate for NLPs.
        Default: ipopt
    mip_solver: Optional[SolverBase]
        A solver interface appropriate for LPs and MILPs.
        Default: gurobi_persistent
    default_bound: float
        All unbounded variables will be given lower and
        upper bounds equal to default_bound.
    max_pwl_refinement_iter: int
        This is the maximum number of iterations used to refine the piecewise linear
        approximation.
    num_pwl_cons_to_refine_per_iter: int
        This is the maximum number of constraints to be refined with additional
        segments in the piecewise linear approximation each iteration.
    aggressive_substitution: bool
        This is passed along to the contrib.piecewise.univariate_nonlinear_decomposition
        transformation.
    skip_initial_nlp_solve: bool
        If True, the initial attempt at solving the NLP without initialization
        will be skipped.
    bounds_tol: float
        Bad things can happen with piecewise linear functions if the value of a
        variable ends up outside of the variable's bounds. This bounds_tol is used
        to ensure that variable values are sufficiently inside of the variable's
        bounds.

    Returns
    -------
    res: pyomo.contrib.solver.common.results.Results
        The results object obtained the last time the nlp_solver was used to
        try and solve the model.
    """
    if nlp_solver is None:
        nlp_solver = _get_solver('ipopt', 'local NLP solver')

    if not skip_initial_nlp_solve:
        res = _try_nlp_solve(nlp, nlp_solver)
        if res.solution_status == SolutionStatus.optimal:
            return res

    if mip_solver is None:
        mip_solver = _get_solver('gurobi_persistent', 'MILP solver')

    orig_var_data = _setup(nlp)

    try:
        res = _initialize_with_piecewise_linear_approximation(
            nlp=nlp,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            default_bound=default_bound,
            max_iter=max_pwl_refinement_iter,
            num_cons_to_refine_per_iter=num_pwl_cons_to_refine_per_iter,
            aggressive_substitution=aggressive_substitution,
            bounds_tol=bounds_tol,
        )
    finally:
        _cleanup(orig_var_data)

    return res


def initialize_with_LP_approximation(
    nlp: BlockData,
    nlp_solver: SolverBase | None = None,
    lp_solver: SolverBase | None = None,
    default_bound: float = 1.0e8,
    use_univariate_nonlinear_decomposition: bool = True,
    aggressive_substitution: bool = False,
    num_samples_per_nonlinear_constraint: int = 100,
    seed=0,
    skip_initial_nlp_solve: bool = False,
) -> Results:
    """
    Attempt to initialize the problem with an LP approximation and
    subsequently solve the model given by ``nlp``.

    Parameters
    ----------
    nlp: BlockData
        The pyomo model to be initialized.
    nlp_solver: Optional[SolverBase]
        A solver interface appropriate for NLPs.
        Default: ipopt
    lp_solver: Optional[SolverBase]
        A solver interface appropriate for LPs.
        Default: gurobi_persistent
    default_bound: float
        Some initialize methods require all nonlinear variables to be bounded.
        For these methods, all unbounded variables will be given lower and
        upper bounds equal to default_bound.
        Needed for the following methods:
          - pwl_approximation
          - lp_approximation
    use_univariate_nonlinear_decomposition: bool
        If False, the transformation
    aggressive_substitution: bool
        This is passed along to the contrib.piecewise.univariate_nonlinear_decomposition
        transformation.
    num_samples_per_nonlinear_constraint: int
        This is the number of random samples used to build the linear least squares
        problem for each nonlinear constraint.
    seed: int | np.random.Generator
        This is used to make the sampling for the linear least squares problems
        deterministic.
    skip_initial_nlp_solve: bool
        If True, the initial attempt at solving the NLP without initialization
        will be skipped.

    Returns
    -------
    res: pyomo.contrib.solver.common.results.Results
        The results object obtained the last time the nlp_solver was used to
        try and solve the model.
    """
    if nlp_solver is None:
        nlp_solver = _get_solver('ipopt', 'local NLP solver')

    if not skip_initial_nlp_solve:
        res = _try_nlp_solve(nlp, nlp_solver)
        if res.solution_status == SolutionStatus.optimal:
            return res

    orig_var_data = _setup(nlp)

    if lp_solver is None:
        lp_solver = _get_solver('gurobi_persistent', 'LP solver')

    try:
        res = _initialize_with_LP_approximation(
            nlp=nlp,
            lp_solver=lp_solver,
            nlp_solver=nlp_solver,
            default_bound=default_bound,
            num_samples=num_samples_per_nonlinear_constraint,
            seed=seed,
            use_univariate_nonlinear_decomposition=use_univariate_nonlinear_decomposition,
            aggressive_substitution=aggressive_substitution,
        )
    finally:
        _cleanup(orig_var_data)

    return res


def initialize_with_global_opt(
    nlp: BlockData,
    nlp_solver: SolverBase | None = None,
    global_solver: SolverBase | None = None,
    skip_initial_nlp_solve: bool = False,
) -> Results:
    """
    Attempt to initialize and subsequently solve the model given by ``nlp``.
    The basic idea is to apply some method to find good initial values for
    the variables and then try to solve the problem with ``nlp_solver``.

    Parameters
    ----------
    nlp: BlockData
        The pyomo model to be initialized.
    nlp_solver: Optional[SolverBase]
        A solver interface appropriate for NLPs.
        Default: ipopt
    global_solver: Optional[SolverBase]
        A solver interface appropriate for global solution of NLPs
        Default: gurobi_direct_minlp
    skip_initial_nlp_solve: bool
        If True, the initial attempt at solving the NLP without initialization
        will be skipped.

    Returns
    -------
    res: pyomo.contrib.solver.common.results.Results
        The results object obtained the last time the nlp_solver was used to
        try and solve the model.
    """
    if nlp_solver is None:
        nlp_solver = _get_solver('ipopt', 'local NLP solver')

    if not skip_initial_nlp_solve:
        res = _try_nlp_solve(nlp, nlp_solver)
        if res.solution_status == SolutionStatus.optimal:
            return res

    orig_var_data = _setup(nlp)

    if global_solver is None:
        global_solver = _get_solver('gurobi_direct_minlp', 'global NLP solver')

    try:
        res = _initialize_with_global_solver(
            nlp=nlp, global_solver=global_solver, nlp_solver=nlp_solver
        )
    finally:
        _cleanup(orig_var_data)

    return res
