# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.core.base.block import BlockData
import pyomo.environ as pe
from pyomo.devel.initialization.bounds.bound_variables import (
    bound_all_nonlinear_variables,
)
from pyomo.devel.initialization.utils import (
    fix_vars_with_equal_bounds,
    shallow_clone,
    get_vars,
)
from pyomo.core.expr.visitor import identify_components
from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression,
)
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
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.repn.util import ExitNodeDispatcher
from pyomo.core.base.var import ScalarVar, VarData
from pyomo.core.base.param import ScalarParam, ParamData
from pyomo.core.base.expression import ScalarExpression, ExpressionData
import math
from pyomo.contrib.solver.common.base import SolverBase
import logging
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.solver.common.results import SolutionStatus

logger = logging.getLogger(__name__)


def _minimize_infeasibility(m):
    m.slacks = pe.VarList()
    m.extra_cons = pe.ConstraintList()

    obj_expr = 0

    found_obj = False
    for obj in m.component_data_objects(pe.Objective, active=True, descend_into=True):
        assert not found_obj
        if obj.sense == pe.minimize:
            obj_expr += 0.1 * obj.expr
        else:
            obj_expr -= 0.1 * obj.expr
        obj.deactivate()
        found_obj = True

    for con in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
        if lb == ub:
            ps = m.slacks.add()
            ns = m.slacks.add()
            ps.setlb(0)
            ns.setlb(0)
            con.set_value(body - lb - ps + ns == 0)
        elif lb is None:
            ps = m.slacks.add()
            ps.setlb(0)
            con.set_value(body - ub - ps <= 0)
        elif ub is None:
            ns = m.slacks.add()
            ns.setlb(0)
            con.set_value(body - lb + ns >= 0)
        else:
            con.deactivate()
            ps = m.slacks.add()
            ns = m.slacks.add()
            ps.setlb(0)
            ns.setlb(0)
            m.extra_cons.add(body - ub - ps <= 0)
            m.extra_cons.add(body - lb + ns >= 0)

    m.slack_obj = pe.Objective(expr=10 * sum(m.slacks.values()) + obj_expr)


def _get_pwl_constraints(
    m: BlockData,
) -> MutableMapping[PiecewiseLinearExpression, List[ConstraintData]]:
    comp_types = set()
    comp_types.add(PiecewiseLinearExpression)
    pwl_expr_to_con_map = ComponentMap()
    con_list = list(
        m.component_data_objects(pe.Constraint, active=True, descend_into=True)
    )
    obj_list = list(
        m.component_data_objects(pe.Objective, active=True, descend_into=True)
    )
    for comp in con_list + obj_list:
        pwl_exprs = list(identify_components(comp.expr, comp_types))
        if not pwl_exprs:
            continue
        assert len(pwl_exprs) == 1
        e = pwl_exprs[0]
        if e not in pwl_expr_to_con_map:
            pwl_expr_to_con_map[e] = []
        pwl_expr_to_con_map[e].append(comp)
    return pwl_expr_to_con_map


def _handle_leaf(node, data):
    return node


def _handle_node(node, data):
    return node.create_node_with_local_data(data)


_handlers = ExitNodeDispatcher()
for t in [float, int, VarData, ScalarVar, ParamData, ScalarParam, NumericConstant]:
    _handlers[t] = _handle_leaf
for t in [
    ProductExpression,
    SumExpression,
    DivisionExpression,
    PowExpression,
    MonomialTermExpression,
    LinearExpression,
    ExpressionData,
    ScalarExpression,
    NegationExpression,
    UnaryFunctionExpression,
    NPV_NegationExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
]:
    _handlers[t] = _handle_node


class _PWLRefinementVisitor(StreamBasedExpressionVisitor):
    def __init__(self, m, pwl_exprs, **kwds):
        self.m = m
        self.pwl_exprs = ComponentSet(pwl_exprs)
        self.substitution = ComponentMap()
        self.named_expr_map = ComponentMap()
        super().__init__(**kwds)

    def exitNode(self, node, data):
        if node in self.named_expr_map:
            return self.named_expr_map[node]
        nt = type(node)
        if nt in _handlers:
            return _handlers[nt](node, data)
        elif nt in native_numeric_types:
            _handlers[nt] = _handle_leaf
            return _handle_leaf(node, data)
        else:
            raise NotImplementedError(f'unrecognized expression type: {nt}')

    def beforeChild(self, node, child, child_idx):
        if child in self.substitution:
            return False, self.substitution[child]

        if child not in self.pwl_exprs:
            return True, None

        old_func = child.pw_linear_function
        _func = old_func._func
        points = list(old_func._points)
        variables = child.args
        var_values = tuple(i.value for i in variables)
        points.append(var_values)
        points.sort()
        if len(points[0]) == 1:
            points = [i[0] for i in points]
        new_func = PiecewiseLinearFunction(points=points, function=_func)
        fname = unique_component_name(
            self.m.auxiliary._pyomo_contrib_nonlinear_to_pwl, 'f'
        )
        setattr(self.m.auxiliary._pyomo_contrib_nonlinear_to_pwl, fname, new_func)
        new_expr = new_func(*variables)
        for v, val in zip(variables, var_values):
            v.set_value(val, skip_validation=True)
        self.named_expr_map[node] = new_expr
        self.substitution[child] = new_expr.expr
        return False, new_expr.expr


def _refine_pwl_approx(
    m,
    pwl_expr_to_con_map: MutableMapping[
        PiecewiseLinearExpression, Sequence[ConstraintData]
    ],
    num_to_refine: int = 5,
):
    violations = []
    for expr in pwl_expr_to_con_map.keys():
        func = expr.pw_linear_function
        var_vals = []
        for v in expr.args:
            if math.isclose(v.lb, v.ub, rel_tol=1e-6, abs_tol=1e-6):
                val = 0.5 * (v.lb + v.ub)
            elif v.value is None:
                val = None
            else:
                val = v.value
                if val <= v.lb + 1e-6 + 1e-6 * abs(v.lb):
                    val += 1e-5
                if val >= v.ub - 1e-6 - 1e-6 * abs(v.ub):
                    val -= 1e-5
            var_vals.append(val)
        # var_vals = tuple(i.value for i in expr.args)
        # for v, val in zip(expr.args, var_vals):
        #     print(f'{str(v):<20}{val:<20.5f}{v.lb:<20.5f}{v.ub:<20.5f}{id(v):<20}')
        if any(i is None for i in var_vals):
            logger.info(f'missing variable values for {expr}')
            continue
        approx_value = func(*var_vals)
        true_value = func._func(*var_vals)
        err = abs(true_value - approx_value)
        violations.append((err, expr))
    violations.sort(key=lambda i: i[0], reverse=True)

    if len(violations) == 0:
        raise RuntimeError(
            'Did not find any piecewise linear functions with variable values'
        )

    tol = 1e-5
    if math.isclose(violations[0][0], 0, abs_tol=tol):
        logger.info('All of the original nonlinear functions are satisfied!')

    violations = [i for i in violations if i[0] > tol]

    for err, expr in violations[:num_to_refine]:
        logger.info(f'refining {expr.pw_linear_function._func.expr} with error {err}')

    funcs_to_refine = ComponentSet(i[1] for i in violations[:num_to_refine])
    visitor = _PWLRefinementVisitor(m, funcs_to_refine)

    for expr in funcs_to_refine:
        for con in pwl_expr_to_con_map[expr]:
            con.set_value(visitor.walk_expression(con.expr))

    for e1, e2 in visitor.substitution.items():
        cons = pwl_expr_to_con_map.pop(e1)
        pwl_expr_to_con_map[e2] = cons


def _initialize_with_piecewise_linear_approximation(
    nlp: BlockData,
    mip_solver: SolverBase,
    nlp_solver: SolverBase,
    default_bound=1.0e8,
    max_iter=100,
    num_cons_to_refine_per_iter=5,
    aggressive_substitution=True,
):
    logger.info('Starting initialization using a piecewise linear approximation')
    pwl = shallow_clone(nlp)
    logger.info('created a shallow clone of the model')

    # first introduce auxiliary variables so that we don't try to
    # approximate any functions of more than two variables
    trans = pe.TransformationFactory(
        'contrib.piecewise.univariate_nonlinear_decomposition'
    )
    trans.apply_to(pwl, aggressive_substitution=aggressive_substitution)
    logger.info('applied the univariate_nonlinear_decomposition transformation')

    # now we need to try to get bounds on all of the nonlinear variables
    bound_all_nonlinear_variables(pwl, default_bound=default_bound)
    logger.info('bounded nonlinear variables')

    # Now, we need to fix variables with equal (or nearly equal) bounds.
    # Otherwise, the PWL transformation complains
    fix_vars_with_equal_bounds(pwl)
    logger.info('fixed variables with equal bounds')

    # now we modify the model by introducing slacks to make sure the PWL
    # approximation is feasible
    # all of the slacks appear linearly, so we don't need to worry about
    # upper bounds for them
    _minimize_infeasibility(pwl)
    logger.info('reformulated model to minimize infeasibility')

    # build the PWL approximation
    trans = pe.TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
    trans.apply_to(pwl, num_points=2, additively_decompose=False)
    logger.info('replaced nonlinear expressions with piecewise linear expressions')

    """
    Now we want to 
    1. solve the PWL approximation
    2. Initialize the NLP to the solution
    3. Try solving the NLP
    4. If the NLP converges => done
    5. If the NLP does not converge, refine the PWL approximation and repeat
    """
    pwl_expr_to_con_map = _get_pwl_constraints(pwl)
    solved = False
    last_nlp_res = None
    for _iter in range(max_iter):
        logger.info(f'PWL initialization: iter {_iter}')

        # PWL transformation (and map the variables)
        orig_vars = list(get_vars(pwl))
        pwl.orig_vars = orig_vars
        trans = pe.TransformationFactory('contrib.piecewise.disaggregated_logarithmic')
        _pwl = trans.create_using(pwl)
        new_vars = _pwl.orig_vars
        del pwl.orig_vars
        del _pwl.orig_vars
        logger.info('applied the disaggregated logarithmic transformation')

        # solve the MILP
        res = mip_solver.solve(
            _pwl, load_solutions=True, raise_exception_on_nonoptimal_result=False
        )
        logger.info(f'solved MILP: {res.solution_status}, {res.termination_condition}')

        # load the variable values back into orig_vars
        for ov, nv in zip(orig_vars, new_vars):
            ov.set_value(nv.value, skip_validation=True)

        # refine the PWL approximation
        _refine_pwl_approx(
            pwl,
            pwl_expr_to_con_map=pwl_expr_to_con_map,
            num_to_refine=num_cons_to_refine_per_iter,
        )
        logger.info('refined PWL approximation')

        # try solving the NLP
        res = nlp_solver.solve(
            nlp, load_solutions=False, raise_exception_on_nonoptimal_result=False
        )
        last_nlp_res = res
        logger.info(f'solved NLP: {res.solution_status}, {res.termination_condition}')
        if res.solution_status in {SolutionStatus.feasible, SolutionStatus.optimal}:
            solved = True
            res.solution_loader.load_vars()
            break

    if not solved:
        logger.warning('initialization was not successful via PWL approximation')

    return last_nlp_res
