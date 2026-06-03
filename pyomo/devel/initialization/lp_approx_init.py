# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import math
from typing import List, MutableMapping, Sequence

import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import numpy as np
from pyomo.common.modeling import unique_component_name
from pyomo.common.numeric_types import native_numeric_types
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression,
)
from pyomo.contrib.piecewise.piecewise_linear_function import PiecewiseLinearFunction
from pyomo.contrib.solver.common.base import SolverBase
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.expression import ExpressionData, ScalarExpression
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.var import ScalarVar, VarData
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.numeric_expr import (
    DivisionExpression,
    LinearExpression,
    MonomialTermExpression,
    NegationExpression,
    NPV_DivisionExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    PowExpression,
    ProductExpression,
    SumExpression,
    UnaryFunctionExpression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor,
    identify_components,
    identify_variables,
)
from pyomo.devel.initialization.bounds.bound_variables import (
    bound_all_nonlinear_variables,
)
from pyomo.devel.initialization.pwl_init import _minimize_infeasibility
from pyomo.devel.initialization.utils import (
    fix_vars_with_equal_bounds,
    get_vars,
    shallow_clone,
)
from pyomo.repn.linear import LinearRepn, LinearRepnVisitor
from pyomo.repn.util import ExitNodeDispatcher
from pyomo.common.dependencies.scipy import stats
from pyomo.common.dependencies import scipy_available

logger = logging.getLogger(__name__)


def _generate_linear_approx(expr, num_samples=100, seed=None):
    vlist = list(identify_variables(expr, include_fixed=False))
    n_vars = len(vlist)
    bnds_list = []
    for v in vlist:
        # the bounds should not be None because we
        # set the bounds to default_bound in
        # bound_all_nonlinear_variables
        lb = v.lb
        ub = v.ub
        bnds_list.append((lb, ub))
    sampler = stats.qmc.LatinHypercube(d=n_vars, seed=seed)
    sample = sampler.random(n=num_samples)
    l_bounds = [i[0] for i in bnds_list]
    u_bounds = [i[1] for i in bnds_list]
    sample = stats.qmc.scale(sample, l_bounds, u_bounds)

    # we have our samples
    # now we want to build the matrix and the right hand side
    # we have a linear coefficient for each variable plus a constant
    n_coefs = n_vars + 1
    A = np.zeros((num_samples, n_coefs), dtype=float)
    b = np.zeros(num_samples, dtype=float)
    A[:, :n_vars] = sample
    A[:, n_vars] = 1
    for sample_ndx in range(num_samples):
        for v, val in zip(vlist, sample[sample_ndx, :]):
            v.value = val
        b[sample_ndx] = pyo.value(expr)
    coefs = np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b)).tolist()

    new_expr = 0
    for c, v in zip(coefs[:n_vars], vlist):
        new_expr += c * v
    new_expr += coefs[-1]
    return new_expr


def _build_lp_approx(nlp: BlockData, num_samples=100, seed=None) -> BlockData:
    lp = pyo.Block(concrete=True)
    lp.cons = pyo.ConstraintList()
    visitor = LinearRepnVisitor(subexpression_cache={})

    seed_handler = np.random.default_rng(seed)

    objs = list(
        nlp.component_data_objects(pyo.Objective, active=True, descend_into=True)
    )
    if objs:
        if len(objs) > 1:
            raise NotImplementedError(
                'lp approximation does not support multiple objectives'
            )
        obj = objs[0]
        repn = visitor.walk_expression(obj)
        if repn.nonlinear is None:
            new_obj_expr = obj.expr
        else:
            linear_part = LinearRepn()
            linear_part.multiplier = 1
            linear_part.constant = repn.constant
            linear_part.linear = repn.linear
            linear_part.nonlinear = None
            new_obj_expr = linear_part.to_expression(visitor=visitor)
            new_obj_expr += _generate_linear_approx(
                repn.nonlinear, num_samples=num_samples, seed=seed_handler.spawn(1)[0]
            )
        lp.obj = pyo.Objective(expr=new_obj_expr, sense=obj.sense)

    for con in nlp.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
        if (lb is None or lb == -float('inf')) and (ub is None or ub == float('inf')):
            continue
        repn = visitor.walk_expression(body)
        if repn.nonlinear is None:
            new_body = body
        else:
            linear_part = LinearRepn()
            linear_part.multiplier = 1
            linear_part.constant = repn.constant
            linear_part.linear = repn.linear
            linear_part.nonlinear = None
            new_body = linear_part.to_expression(visitor=visitor)
            new_body += _generate_linear_approx(
                repn.nonlinear, num_samples=num_samples, seed=seed_handler.spawn(1)[0]
            )
        if lb == ub:
            # we checked for unbounded constraints above
            lp.cons.add(new_body == lb)
        else:
            lp.cons.add((lb, new_body, ub))
    return lp


def _initialize_with_LP_approximation(
    nlp: BlockData,
    lp_solver: SolverBase,
    nlp_solver: SolverBase,
    default_bound=1.0e8,
    num_samples=100,
    seed=None,
    use_univariate_nonlinear_decomposition: bool = True,
    aggressive_substitution: bool = False,
):
    orig_nlp = nlp
    logger.info('Starting initialization using a linear programming approximation')
    nlp = shallow_clone(nlp)
    logger.info('created a shallow clone of the model')

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

    # Now, we need to fix variables with equal (or nearly equal) bounds.
    # Otherwise, the PWL transformation complains
    fix_vars_with_equal_bounds(nlp)
    logger.info('fixed variables with equal bounds')

    # build the LP approximation
    lp = _build_lp_approx(nlp, num_samples=num_samples, seed=seed)
    logger.info('replaced nonlinear expressions with linear approximations')

    # now we modify the model by introducing slacks to make sure the LP
    # approximation is feasible
    _minimize_infeasibility(lp)
    logger.info('reformulated model to minimize infeasibility')

    # solve the LP
    lp_res = lp_solver.solve(
        lp, load_solutions=True, raise_exception_on_nonoptimal_result=False
    )
    logger.info(f'solved LP: {lp_res.solution_status}, {lp_res.termination_condition}')

    # try solving the NLP
    nlp_res = nlp_solver.solve(
        orig_nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False
    )
    logger.info(
        f'solved NLP: {nlp_res.solution_status}, {nlp_res.termination_condition}'
    )

    if nlp_res.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
        nlp_res.solution_loader.load_vars()
    else:
        logger.warning('initialization was not successful via LP approximation')

    return nlp_res
