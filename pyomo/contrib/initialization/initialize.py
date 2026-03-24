from typing import Optional
from pyomo.core.base.block import BlockData
from enum import Enum
from pyomo.contrib.initialization.utils import get_vars, shallow_clone
from pyomo.common.collections import ComponentMap
from pyomo.contrib.initialization.pwl_init import _initialize_with_piecewise_linear_approximation
from pyomo.contrib.initialization.lp_approx_init import _initialize_with_LP_approximation
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.initialization.global_init import _initialize_with_global_solver
from pyomo.contrib.solver.common.factory import SolverFactory
import logging


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
    mip_solver: Optional[SolverBase] = None,
    nlp_solver: Optional[SolverBase] = None,
    global_solver: Optional[SolverBase] = None,
    method: InitializationMethod = InitializationMethod.global_opt,
    default_bound=1.0e8,
    max_pwl_refinement_iter=100,
    num_pwl_cons_to_refine_per_iter=5,
):
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
