from typing import Optional
from pyomo.core.base.block import BlockData
from enum import Enum
from pyomo.devel.initialization.utils import get_vars, shallow_clone
# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.collections import ComponentMap
from pyomo.devel.initialization.pwl_init import _initialize_with_piecewise_linear_approximation
from pyomo.devel.initialization.lp_approx_init import _initialize_with_LP_approximation
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.devel.initialization.global_init import _initialize_with_global_solver
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import Results
import logging
from pyomo.contrib.solver.common.results import SolutionStatus


logger = logging.getLogger(__name__)


class InitializationMethod(Enum):
    pwl_approximation = "pwl_approximation"
    lp_approximation = "lp_approximation"
    global_opt = "global_opt"


def _get_solver(sname, reason):
    opt = SolverFactory(sname)
    if opt.available():
        logger.info(f'Using {sname} for {reason} because a solver was not specified')
    else:
        raise RuntimeError(f'No solver was specified for {reason} and the default ({sname}) is not available')
    return opt


def initialize_nlp(
    nlp: BlockData, 
    nlp_solver: SolverBase,
    mip_solver: Optional[SolverBase] = None,
    global_solver: Optional[SolverBase] = None,
    method: InitializationMethod = InitializationMethod.global_opt,
    default_bound: float = 1.0e8,
    max_pwl_refinement_iter: int = 100,
    num_pwl_cons_to_refine_per_iter: int = 5,
) -> Results:
    """
    Attempt to initialize and subsequently solve the model given by ``nlp``.
    The basic idea is to apply some method to find good initial values for 
    the variables and then try to solve the problem with ``nlp_solver``.

    Parameters
    ----------
    nlp: BlockData
        The pyomo model to be initialized.
    mip_solver: Optional[SolverBase]
        A solver interface appropriate for LPs and MILPs.
        Needed for the following methods:
          - pwl_approximation
          - lp_approximation
        Default: gurobi_persistent
    nlp_solver: Optional[SolverBase]
        A solver interface appropriate for NLPs.
        Default: ipopt
    global_solver: Optional[SolverBase]
        A solver interface appropriate for global solution of NLPs
        Default: gurobi_direct_minlp
    method: InitializationMethod
        The method used to initialize the model.
    default_bound: float
        Some initialize methods require all nonlinear variables to be bounded.
        For these methods, all unbounded variables will be given lower and 
        upper bounds equal to default_bound.
        Needed for the following methods:
          - pwl_approximation
          - lp_approximation
    max_pwl_refinement_iter: int
        Only used when method = InitializationMethod.pwl_approximation. This is 
        the maximum number of iterations used to refine the piecewise linear
        approximation.
    num_pwl_cons_to_refine_per_iter: int
        Only used when method = InitializationMethod.pwl_approximation. This is
        the maximum number of constraints to be refined with additional 
        segments in the piecewise linear approximation each iteration.

    Returns
    -------
    res: Results
        The results object obtained the last time the nlp_solver was used to 
        try and solve the model.
    """
    # in all cases, try to solve the nlp before doing extra work
    res = nlp_solver.solve(nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False)
    logger.info(f'solved NLP: {res.solution_status}, {res.termination_condition}')

    if res.solution_status == SolutionStatus.optimal:
        res.solution_loader.load_vars()
        logger.info('NLP solved without any initialization')
        return res

    # get all variable bounds, domains, etc. to restore them later
    orig_vars = get_vars(nlp)
    orig_var_data = ComponentMap(
        (v, (v.lower, v.upper, v.domain, v.fixed, v.value)) for v in orig_vars
    )

    # run the initialization
    if method == InitializationMethod.pwl_approximation:
        if mip_solver is None:
            mip_solver = _get_solver('gurobi_persistent', 'MILP solver')
        if nlp_solver is None:
            nlp_solver = _get_solver('ipopt', 'local NLP solver')
        res = _initialize_with_piecewise_linear_approximation(
            nlp=nlp,
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            default_bound=default_bound,
            max_iter=max_pwl_refinement_iter,
            num_cons_to_refine_per_iter=num_pwl_cons_to_refine_per_iter,
        )
    elif method == InitializationMethod.lp_approximation:
        if mip_solver is None:
            mip_solver = _get_solver('gurobi_persistent', 'MILP solver')
        if nlp_solver is None:
            nlp_solver = _get_solver('ipopt', 'local NLP solver')
        res = _initialize_with_LP_approximation(
            nlp=nlp,
            lp_solver=mip_solver,
            nlp_solver=nlp_solver,
        )
    elif method == InitializationMethod.global_opt:
        if global_solver is None:
            global_solver = _get_solver('gurobi_direct_minlp', 'global NLP solver')
        if nlp_solver is None:
            nlp_solver = _get_solver('ipopt', 'local NLP solver')
        res = _initialize_with_global_solver(nlp=nlp, global_solver=global_solver, nlp_solver=nlp_solver)
    else:
        raise ValueError(f'unexpected initialization method: {method}')    

    # restore variable bounds, domain, etc.
    for v, (lb, ub, domain, fixed, value) in orig_var_data.items():
        v.setlb(lb)
        v.setub(ub)
        v.domain = domain
        if fixed:
            assert v.value == value
            assert v.fixed
        else:
            v.unfix()

    return res
