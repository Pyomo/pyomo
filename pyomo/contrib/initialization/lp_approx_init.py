from pyomo.core.base.block import BlockData
import pyomo.environ as pe
from pyomo.contrib.initialization.bounds.bound_variables import bound_all_nonlinear_variables
from pyomo.contrib.initialization.utils import fix_vars_with_equal_bounds, shallow_clone, get_vars
from pyomo.core.expr.visitor import identify_components
from pyomo.contrib.piecewise.piecewise_linear_expression import PiecewiseLinearExpression
from pyomo.contrib.piecewise.piecewise_linear_function import PiecewiseLinearFunction
from pyomo.common.collections import ComponentMap, ComponentSet
from typing import MutableMapping, Sequence, List
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    PowExpression,
    ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    UnaryFunctionExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.expr.relational_expr import EqualityExpression, InequalityExpression, RangedExpression
from pyomo.repn.util import ExitNodeDispatcher
from pyomo.core.base.var import ScalarVar, VarData
from pyomo.core.base.param import ScalarParam, ParamData
from pyomo.core.base.expression import ScalarExpression, ExpressionData
import math
from pyomo.contrib.solver.common.base import SolverBase
import logging
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.initialization.pwl_init import _minimize_infeasibility
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.repn.linear import LinearRepnVisitor, LinearRepn
from pyomo.core.expr.visitor import identify_variables
import numpy as np
from scipy.stats import qmc


logger = logging.getLogger(__name__)


def _replace_expression_with_linear_approx(expr, num_samples=100):
    vset = ComponentSet(identify_variables(expr, include_fixed=False))
    vlist = list(vset)
    n_vars = len(vlist)
    bnds_list = []
    for v in vlist:
        if v.lb is None:
            lb = -1e6
        else:
            lb = v.lb
        if v.ub is None:
            ub = 1e6
        else:
            ub = v.ub
        bnds_list.append((lb, ub))
    sampler = qmc.LatinHypercube(d=n_vars)
    sample = sampler.random(n=num_samples)
    l_bounds = [i[0] for i in bnds_list]
    u_bounds = [i[1] for i in bnds_list]
    sample = qmc.scale(sample, l_bounds, u_bounds)

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
            v.value = float(val)
        b[sample_ndx] = pe.value(expr)
    coefs = np.linalg.solve(A.transpose().dot(A), A.transpose().dot(b))
    coefs = [float(i) for i in coefs]

    new_expr = 0
    for c, v in zip(coefs[:n_vars], vlist):
        new_expr += c * v
    new_expr += coefs[-1]
    return new_expr


def _build_lp_approx(nlp: BlockData) -> BlockData:
    lp = pe.Block(concrete=True)
    lp.cons = pe.ConstraintList()
    visitor = LinearRepnVisitor(subexpression_cache={})

    objs = list(nlp.component_data_objects(pe.Objective, active=True, descend_into=True))
    if objs:
        if len(objs) > 1:
            raise NotImplementedError('lp approximation does not support multiple objectives')
        obj = objs[0]
        repn = visitor.walk_expression(obj)
        assert repn.multiplier == 1
        linear_part = LinearRepn()
        linear_part.multiplier = 1
        linear_part.constant = repn.constant
        linear_part.linear = repn.linear
        linear_part.nonlinear = None
        new_obj_expr = linear_part.to_expression(visitor=visitor)
        if repn.nonlinear is not None:
            replacement = _replace_expression_with_linear_approx(repn.nonlinear)
            new_obj_expr += replacement
        lp.obj = pe.Objective(expr=new_obj_expr, sense=obj.sense)

    for con in nlp.component_data_objects(pe.Constraint, active=True, descend_into=True):
        lb, body, ub = con.to_bounded_expression()
        repn = visitor.walk_expression(body)
        assert repn.multiplier == 1
        linear_part = LinearRepn()
        linear_part.multiplier = 1
        linear_part.constant = repn.constant
        linear_part.linear = repn.linear
        linear_part.nonlinear = None
        new_body = linear_part.to_expression(visitor=visitor)
        if repn.nonlinear is not None:
            replacement = _replace_expression_with_linear_approx(repn.nonlinear)
            new_body += replacement
        if lb == ub:
            lp.cons.add(new_body == lb)
        else:
            lp.cons.add((lb, new_body, ub))
    return lp


def _initialize_with_LP_approximation(
    nlp: BlockData,
    lp_solver: SolverBase,
    nlp_solver: SolverBase,
):
    orig_nlp = nlp
    logger.info('Starting initialization using a linear programming approximation')
    nlp = shallow_clone(nlp)
    logger.info('created a shallow clone of the model')

    # first introduce auxiliary variables so that we don't try to 
    # approximate any functions of more than two variables
    trans = pe.TransformationFactory('contrib.piecewise.univariate_nonlinear_decomposition')
    trans.apply_to(nlp, aggressive_substitution=False)
    logger.info('applied the univariate_nonlinear_decomposition transformation')

    # let's see if we can get bounds on the nonlinear variables
    fbbt(nlp)
    logger.info('ran FBBT')

    # Now, we need to fix variables with equal (or nearly equal) bounds.
    # Otherwise, the PWL transformation complains
    fix_vars_with_equal_bounds(nlp)
    logger.info('fixed variables with equal bounds')

    # now we modify the model by introducing slacks to make sure the LP
    # approximatin is feasible
    _minimize_infeasibility(nlp)
    logger.info('reformulated model to minimize infeasibility')

    # build the LP approximation
    lp = _build_lp_approx(nlp)
    logger.info('replaced nonlinear expressions with linear approximations')

    # solve the LP
    lp_res = lp_solver.solve(lp, load_solutions=True, raise_exception_on_nonoptimal_result=False)
    logger.info(f'solved LP: {lp_res.solution_status}, {lp_res.termination_condition}')

    # try solving the NLP
    nlp_res = nlp_solver.solve(orig_nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False)
    logger.info(f'solved NLP: {nlp_res.solution_status}, {nlp_res.termination_condition}')
    if nlp_res.incumbent_objective is not None:
        nlp_res.solution_loader.load_vars()
    else:
        raise RuntimeError('no feasible solution found')

    return nlp_res
