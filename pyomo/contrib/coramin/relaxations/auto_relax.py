import pyomo.environ as pe
from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import ExpressionValueVisitor
from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types, value, NumericValue, is_fixed, polynomial_degree, is_constant,
    native_numeric_types
)
from pyomo.core.expr.numeric_expr import ExpressionBase
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr, fbbt
import math
from pyomo.core.base.constraint import Constraint
import logging
from .relaxations_base import BaseRelaxationData
from .univariate import PWUnivariateRelaxation, PWXSquaredRelaxation, PWCosRelaxation, PWSinRelaxation, PWArctanRelaxation
from .mccormick import PWMcCormickRelaxation
from .multivariate import MultivariateRelaxation
from .alphabb import AlphaBBRelaxation
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape, Effort, EigenValueBounder
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, SimpleExpression
from coramin.relaxations.iterators import nonrelaxation_component_data_objects
from pyomo.contrib import appsi
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.contrib.fbbt import interval
from pyomo.core.expr.compare import convert_expression_to_prefix_notation
from .split_expr import split_expr
from coramin.utils.pyomo_utils import simplify_expr, active_vars
from .hessian import Hessian
from typing import MutableMapping, Tuple, Union, Optional
from pyomo.core.base.block import _BlockData
from .iterators import relaxation_data_objects


logger = logging.getLogger(__name__)


class Hashable:
    def __init__(self, *args):
        entries = list()
        for i in args:
            itype = type(i)
            if itype is tuple or itype in nonpyomo_leaf_types:
                entries.append(i)
            elif isinstance(i, NumericValue):
                entries.append(id(i))
            else:
                raise NotImplementedError(
                    f'unexpected entry: {str(i)}')
        self.entries = entries
        self.hashable_entries = tuple(entries)

    def __eq__(self, other):
        if isinstance(other, Hashable):
            return self.entries == other.entries
        return False

    def __hash__(self):
        return hash(self.hashable_entries)


class RelaxationException(Exception):
    pass


class RelaxationCounter(object):
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def __str__(self):
        return str(self.count)


def compute_float_bounds_on_expr(expr):
    lb, ub = compute_bounds_on_expr(expr)
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf

    return lb, ub


def replace_sub_expression_with_aux_var(arg, parent_block):
    if type(arg) in nonpyomo_leaf_types:
        return arg
    elif arg.is_expression_type():
        _var = parent_block.aux_vars.add()
        _con = parent_block.aux_cons.add(_var == arg)
        fbbt(_con)
        return _var
    else:
        return arg


def _get_aux_var(parent_block, expr):
    _aux_var = parent_block.aux_vars.add()
    lb, ub = compute_bounds_on_expr(expr)
    _aux_var.setlb(lb)
    _aux_var.setub(ub)
    try:
        expr_value = pe.value(expr, exception=False)
    except ArithmeticError:
        expr_value = None
    if expr_value is not None and pe.value(_aux_var, exception=False) is None:
        _aux_var.set_value(expr_value, skip_validation=True)
    return _aux_var


def _relax_leaf_to_root_ProductExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg1, arg2 = values

    # The purpose of the next bit of code is to find common quadratic terms. For example, suppose we are relaxing
    # a model with the following two constraints:
    #
    # w1 - x*y = 0
    # w2 + 3*x*y = 0
    #
    # we want to end up with
    #
    # w1 - aux1 = 0
    # w2 + 3*aux1 = 0
    # aux1 = x*y
    #
    # rather than
    #
    # w1 - aux1 = 0
    # w2 + 3*aux2 = 0
    # aux1 = x*y
    # aux2 = x*y
    #

    h1 = Hashable(arg1, arg2, 'mul')
    h2 = Hashable(arg2, arg1, 'mul')
    if h1 in aux_var_map or h2 in aux_var_map:
        if h1 in aux_var_map:
            _aux_var, relaxation = aux_var_map[h1]
        else:
            _aux_var, relaxation = aux_var_map[h2]
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        res = _aux_var
        degree_map[res] = 1
    else:
        degree_1 = degree_map[arg1]
        degree_2 = degree_map[arg2]
        if degree_1 == 0:
            res = arg1 * arg2
            degree_map[res] = degree_2
        elif degree_2 == 0:
            res = arg2 * arg1
            degree_map[res] = degree_1
        elif arg1.__class__ == numeric_expr.MonomialTermExpression or arg2.__class__ == numeric_expr.MonomialTermExpression:
            if arg1.__class__ == numeric_expr.MonomialTermExpression:
                coef1, arg1 = arg1.args
            else:
                coef1 = 1
            if arg2.__class__ == numeric_expr.MonomialTermExpression:
                coef2, arg2 = arg2.args
            else:
                coef2 = 1
            coef = coef1 * coef2
            _new_relaxation_side_map = ComponentMap()
            _reformulated = coef * (arg1 * arg2)
            _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
            res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                              relaxation_side_map=_new_relaxation_side_map, counter=counter, degree_map=degree_map)
            degree_map[res] = 1
        elif arg1 is arg2:
            # reformulate arg1 * arg2 as arg1**2
            _new_relaxation_side_map = ComponentMap()
            _reformulated = arg1**2
            _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
            res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                              relaxation_side_map=_new_relaxation_side_map, counter=counter, degree_map=degree_map)
            degree_map[res] = 1
        else:
            _aux_var = _get_aux_var(parent_block, arg1 * arg2)
            arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
            arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
            relaxation_side = relaxation_side_map[node]
            relaxation = PWMcCormickRelaxation()
            relaxation.set_input(x1=arg1, x2=arg2, aux_var=_aux_var, relaxation_side=relaxation_side)
            aux_var_map[h1] = (_aux_var, relaxation)
            setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
            counter.increment()
            res = _aux_var
            degree_map[res] = 1
    return res


def _relax_leaf_to_root_DivisionExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg1, arg2 = values
    h1 = Hashable(arg1, arg2, 'div')
    if arg1.__class__ == numeric_expr.MonomialTermExpression:
        coef1, arg1 = arg1.args
    else:
        coef1 = 1
    if arg2.__class__ == numeric_expr.MonomialTermExpression:
        coef2, arg2 = arg2.args
    else:
        coef2 = 1
    coef = coef1/coef2
    degree_1 = degree_map[arg1]
    degree_2 = degree_map[arg2]

    if degree_2 == 0:
        res = (coef / arg2) * arg1
        degree_map[res] = degree_1
        return res
    elif h1 in aux_var_map:
        _aux_var, relaxation = aux_var_map[h1]
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        res = coef * _aux_var
        degree_map[_aux_var] = 1
        degree_map[res] = 1
        return res
    elif degree_1 == 0:
        h2 = Hashable(arg2, 'reciprocal')
        if h2 in aux_var_map:
            _aux_var, relaxation = aux_var_map[h2]
            relaxation_side = relaxation_side_map[node]
            if relaxation_side != relaxation.relaxation_side:
                relaxation.relaxation_side = RelaxationSide.BOTH
            res = coef * arg1 * _aux_var
            degree_map[_aux_var] = 1
            degree_map[res] = 1
            return res
        else:
            _aux_var = _get_aux_var(parent_block, 1/arg2)
            arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
            relaxation_side = relaxation_side_map[node]
            degree_map[_aux_var] = 1
            if compute_float_bounds_on_expr(arg2)[0] > 0:
                relaxation = PWUnivariateRelaxation()
                relaxation.set_input(x=arg2, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=1/arg2,
                                     shape=FunctionShape.CONVEX)
            elif compute_float_bounds_on_expr(arg2)[1] < 0:
                relaxation = PWUnivariateRelaxation()
                relaxation.set_input(x=arg2, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=1/arg2,
                                     shape=FunctionShape.CONCAVE)
            else:
                _one = parent_block.aux_vars.add()
                _one.fix(1)
                relaxation = PWMcCormickRelaxation()
                relaxation.set_input(x1=arg2, x2=_aux_var, aux_var=_one, relaxation_side=relaxation_side)
            aux_var_map[h2] = (_aux_var, relaxation)
            setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
            counter.increment()
            res = coef * arg1 * _aux_var
            degree_map[res] = 1
            return res
    else:
        _aux_var = _get_aux_var(parent_block, arg1 / arg2)
        arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
        arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
        relaxation_side = relaxation_side_map[node]
        arg2_lb, arg2_ub = compute_float_bounds_on_expr(arg2)
        if arg2_lb >= 0:
            if relaxation_side == RelaxationSide.UNDER:
                relaxation_side = RelaxationSide.OVER
            elif relaxation_side == RelaxationSide.OVER:
                relaxation_side = RelaxationSide.UNDER
            else:
                assert relaxation_side == RelaxationSide.BOTH
        elif arg2_ub <= 0:
            pass
        else:
            relaxation_side = RelaxationSide.BOTH
        relaxation = PWMcCormickRelaxation()
        relaxation.set_input(x1=arg2, x2=_aux_var, aux_var=arg1, relaxation_side=relaxation_side)
        aux_var_map[h1] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        res = coef * _aux_var
        degree_map[_aux_var] = 1
        degree_map[res] = 1
        return res


def _relax_quadratic(arg1, aux_var_map, relaxation_side, degree_map, parent_block, counter):
    _aux_var = _get_aux_var(parent_block, arg1**2)
    arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
    degree_map[_aux_var] = 1
    relaxation = PWXSquaredRelaxation()
    relaxation.set_input(x=arg1, aux_var=_aux_var, relaxation_side=relaxation_side)
    aux_var_map[Hashable(arg1, 2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_convex_pow(arg1, arg2, aux_var_map, relaxation_side, degree_map, parent_block, counter, swap=False):
    _aux_var = _get_aux_var(parent_block, arg1**arg2)
    if swap:
        arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
        _x = arg2
    else:
        arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
        _x = arg1
    degree_map[_aux_var] = 1
    relaxation = PWUnivariateRelaxation()
    relaxation.set_input(x=_x, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=arg1 ** arg2,
                         shape=FunctionShape.CONVEX)
    aux_var_map[Hashable(arg1, arg2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_concave_pow(arg1, arg2, aux_var_map, relaxation_side, degree_map, parent_block, counter):
    _aux_var = _get_aux_var(parent_block, arg1 ** arg2)
    arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
    degree_map[_aux_var] = 1
    relaxation = PWUnivariateRelaxation()
    relaxation.set_input(x=arg1, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=arg1 ** arg2,
                         shape=FunctionShape.CONCAVE)
    aux_var_map[Hashable(arg1, arg2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_leaf_to_root_PowExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg1, arg2 = values
    h = Hashable(arg1, arg2, 'pow')
    if h in aux_var_map:
        _aux_var, relaxation = aux_var_map[h]
        if relaxation_side_map[node] != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        degree1 = degree_map[arg1]
        degree2 = degree_map[arg2]
        if degree2 == 0:
            if degree1 == 0:
                res = arg1 ** arg2
                degree_map[res] = 0
                return res
            if not is_constant(arg2):
                logger.warning('Only constant exponents are supported: ' + str(arg1**arg2) + '\nReplacing ' + str(arg2) + ' with its value.')
            arg2 = pe.value(arg2)
            if arg2 == 1:
                return arg1
            elif arg2 == 0:
                res = 1
                degree_map[res] = 0
                return res
            elif arg2 == 2:
                return _relax_quadratic(arg1=arg1, aux_var_map=aux_var_map, relaxation_side=relaxation_side_map[node],
                                        degree_map=degree_map, parent_block=parent_block, counter=counter)
            elif arg2 >= 0:
                if arg2 == round(arg2):
                    if arg2 % 2 == 0 or compute_float_bounds_on_expr(arg1)[0] >= 0:
                        return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                 relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                 parent_block=parent_block, counter=counter)
                    elif compute_float_bounds_on_expr(arg1)[1] <= 0:
                        return _relax_concave_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                  relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                  parent_block=parent_block, counter=counter)
                    else:  # reformulate arg1 ** arg2 as arg1 * arg1 ** (arg2 - 1)
                        _new_relaxation_side_map = ComponentMap()
                        _reformulated = arg1 * arg1 ** (arg2 - 1)
                        _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
                        res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                                          relaxation_side_map=_new_relaxation_side_map, counter=counter,
                                          degree_map=degree_map)
                        degree_map[res] = 1
                        return res
                else:
                    if arg2 < 1:
                        return _relax_concave_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                  relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                  parent_block=parent_block, counter=counter)
                    else:
                        return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                 relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                 parent_block=parent_block, counter=counter)
            else:
                if arg2 == round(arg2):
                    if compute_float_bounds_on_expr(arg1)[0] >= 0:
                        return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                 relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                 parent_block=parent_block, counter=counter)
                    elif compute_float_bounds_on_expr(arg1)[1] <= 0:
                        if arg2 % 2 == 0:
                            return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                     relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                     parent_block=parent_block, counter=counter)
                        else:
                            return _relax_concave_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                                      relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                                      parent_block=parent_block, counter=counter)
                    else:
                        # reformulate arg1 ** arg2 as 1 / arg1 ** (-arg2)
                        _new_relaxation_side_map = ComponentMap()
                        _reformulated = 1 / (arg1 ** (-arg2))
                        _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
                        res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                                          relaxation_side_map=_new_relaxation_side_map, counter=counter,
                                          degree_map=degree_map)
                        degree_map[res] = 1
                        return res
                else:
                    assert compute_float_bounds_on_expr(arg1)[0] >= 0
                    return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                             relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                             parent_block=parent_block, counter=counter)
        elif degree1 == 0:
            if not is_constant(arg1):
                logger.warning('Found {0} raised to a variable power. However, {0} does not appear to be constant (maybe '
                               'it is or depends on a mutable Param?). Replacing {0} with its value.'.format(str(arg1)))
                arg1 = pe.value(arg1)
            if arg1 < 0:
                raise ValueError('Cannot raise a negative base to a variable exponent: ' + str(arg1**arg2))
            return _relax_convex_pow(arg1=arg1, arg2=arg2, aux_var_map=aux_var_map,
                                     relaxation_side=relaxation_side_map[node], degree_map=degree_map,
                                     parent_block=parent_block, counter=counter, swap=True)
        else:
            assert compute_float_bounds_on_expr(arg1)[0] >= 0
            _new_relaxation_side_map = ComponentMap()
            _reformulated = pe.exp(arg2 * pe.log(arg1))
            _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
            res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                              relaxation_side_map=_new_relaxation_side_map, counter=counter,
                              degree_map=degree_map)
            degree_map[res] = 1
            return res


def _relax_leaf_to_root_SumExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    res = sum(values)
    degree_map[res] = max([degree_map[arg] for arg in values])
    return res


def _relax_leaf_to_root_NegationExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    res = -arg
    degree_map[res] = degree_map[arg]
    return res


def _relax_leaf_to_root_sqrt(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    _new_relaxation_side_map = ComponentMap()
    _reformulated = arg**0.5
    _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
    res = _relax_expr(expr=_reformulated, aux_var_map=aux_var_map, parent_block=parent_block,
                      relaxation_side_map=_new_relaxation_side_map, counter=counter,
                      degree_map=degree_map)
    degree_map[res] = 1
    return res


def _relax_leaf_to_root_exp(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.exp(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'exp') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'exp']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.exp(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWUnivariateRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=pe.exp(arg),
                             shape=FunctionShape.CONVEX)
        aux_var_map[id(arg), 'exp'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_log(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.exp(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'log') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'log']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.log(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWUnivariateRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=pe.log(arg),
                             shape=FunctionShape.CONCAVE)
        aux_var_map[id(arg), 'log'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_log10(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.exp(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'log10') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'log10']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.log10(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWUnivariateRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side, f_x_expr=pe.log10(arg),
                             shape=FunctionShape.CONCAVE)
        aux_var_map[id(arg), 'log10'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_sin(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.sin(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'sin') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'sin']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.sin(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWSinRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side)
        aux_var_map[id(arg), 'sin'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_cos(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.cos(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'cos') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'cos']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.cos(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWCosRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side)
        aux_var_map[id(arg), 'cos'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_arctan(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.atan(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'arctan') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'arctan']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.atan(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1
        relaxation = PWArctanRelaxation()
        relaxation.set_input(x=arg, aux_var=_aux_var, relaxation_side=relaxation_side)
        aux_var_map[id(arg), 'arctan'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_tan(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    degree = degree_map[arg]
    if degree == 0:
        res = pe.tan(arg)
        degree_map[res] = 0
        return res
    elif (id(arg), 'tan') in aux_var_map:
        _aux_var, relaxation = aux_var_map[id(arg), 'tan']
        relaxation_side = relaxation_side_map[node]
        if relaxation_side != relaxation.relaxation_side:
            relaxation.relaxation_side = RelaxationSide.BOTH
        degree_map[_aux_var] = 1
        return _aux_var
    else:
        _aux_var = _get_aux_var(parent_block, pe.tan(arg))
        arg = replace_sub_expression_with_aux_var(arg, parent_block)
        relaxation_side = relaxation_side_map[node]
        degree_map[_aux_var] = 1

        if arg.lb >=0 and arg.ub <= math.pi/2:
            relaxation = PWUnivariateRelaxation()
            relaxation.set_input(
                x=arg, aux_var=_aux_var, shape=FunctionShape.CONVEX,
                f_x_expr=pe.tan(arg), relaxation_side=relaxation_side
            )
        elif arg.lb >= -math.pi/2 and arg.ub <= 0:
            relaxation = PWUnivariateRelaxation()
            relaxation.set_input(
                x=arg, aux_var=_aux_var, shape=FunctionShape.CONCAVE,
                f_x_expr=pe.tan(arg), relaxation_side=relaxation_side
            )
        else:
            raise NotImplementedError('Use alpha-BB here')
        aux_var_map[id(arg), 'tan'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        return _aux_var


_unary_leaf_to_root_map = dict()
_unary_leaf_to_root_map['exp'] = _relax_leaf_to_root_exp
_unary_leaf_to_root_map['log'] = _relax_leaf_to_root_log
_unary_leaf_to_root_map['log10'] = _relax_leaf_to_root_log10
_unary_leaf_to_root_map['sin'] = _relax_leaf_to_root_sin
_unary_leaf_to_root_map['cos'] = _relax_leaf_to_root_cos
_unary_leaf_to_root_map['atan'] = _relax_leaf_to_root_arctan
_unary_leaf_to_root_map['sqrt'] = _relax_leaf_to_root_sqrt
_unary_leaf_to_root_map['tan'] = _relax_leaf_to_root_tan


def _relax_leaf_to_root_UnaryFunctionExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    if node.getname() in _unary_leaf_to_root_map:
        return _unary_leaf_to_root_map[node.getname()](node=node, values=values, aux_var_map=aux_var_map,
                                                       degree_map=degree_map, parent_block=parent_block,
                                                       relaxation_side_map=relaxation_side_map, counter=counter)
    else:
        raise NotImplementedError('Cannot automatically relax ' + str(node))


def _relax_leaf_to_root_GeneralExpression(node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter):
    arg = values[0]
    return arg


_relax_leaf_to_root_map = dict()
_relax_leaf_to_root_map[numeric_expr.ProductExpression] = _relax_leaf_to_root_ProductExpression
_relax_leaf_to_root_map[numeric_expr.SumExpression] = _relax_leaf_to_root_SumExpression
_relax_leaf_to_root_map[numeric_expr.LinearExpression] = _relax_leaf_to_root_SumExpression
_relax_leaf_to_root_map[numeric_expr.MonomialTermExpression] = _relax_leaf_to_root_ProductExpression
_relax_leaf_to_root_map[numeric_expr.NegationExpression] = _relax_leaf_to_root_NegationExpression
_relax_leaf_to_root_map[numeric_expr.PowExpression] = _relax_leaf_to_root_PowExpression
_relax_leaf_to_root_map[numeric_expr.DivisionExpression] = _relax_leaf_to_root_DivisionExpression
_relax_leaf_to_root_map[numeric_expr.UnaryFunctionExpression] = _relax_leaf_to_root_UnaryFunctionExpression
_relax_leaf_to_root_map[numeric_expr.NPV_ProductExpression] = _relax_leaf_to_root_ProductExpression
_relax_leaf_to_root_map[numeric_expr.NPV_SumExpression] = _relax_leaf_to_root_SumExpression
_relax_leaf_to_root_map[numeric_expr.NPV_NegationExpression] = _relax_leaf_to_root_NegationExpression
_relax_leaf_to_root_map[numeric_expr.NPV_PowExpression] = _relax_leaf_to_root_PowExpression
_relax_leaf_to_root_map[numeric_expr.NPV_DivisionExpression] = _relax_leaf_to_root_DivisionExpression
_relax_leaf_to_root_map[numeric_expr.NPV_UnaryFunctionExpression] = _relax_leaf_to_root_UnaryFunctionExpression
_relax_leaf_to_root_map[_GeneralExpressionData] = _relax_leaf_to_root_GeneralExpression
_relax_leaf_to_root_map[SimpleExpression] = _relax_leaf_to_root_GeneralExpression


def _relax_root_to_leaf_ProductExpression(node, relaxation_side_map):
    arg1, arg2 = node.args
    if is_fixed(arg1):
        relaxation_side_map[arg1] = RelaxationSide.BOTH
        if isinstance(arg1, numeric_expr.ProductExpression):  # see Pyomo issue #1147
            arg1_arg1 = arg1.args[0]
            arg1_arg2 = arg1.args[1]
            try:
                arg1_arg1_val = pe.value(arg1_arg1)
            except ValueError:
                arg1_arg1_val = None
            try:
                arg1_arg2_val = pe.value(arg1_arg2)
            except ValueError:
                arg1_arg2_val = None
            if arg1_arg1_val == 0 or arg1_arg2_val == 0:
                arg1_val = 0
            else:
                arg1_val = pe.value(arg1)
        else:
            arg1_val = pe.value(arg1)
        if arg1_val >= 0:
            relaxation_side_map[arg2] = relaxation_side_map[node]
        else:
            if relaxation_side_map[node] == RelaxationSide.UNDER:
                relaxation_side_map[arg2] = RelaxationSide.OVER
            elif relaxation_side_map[node] == RelaxationSide.OVER:
                relaxation_side_map[arg2] = RelaxationSide.UNDER
            else:
                relaxation_side_map[arg2] = RelaxationSide.BOTH
    elif is_fixed(arg2):
        relaxation_side_map[arg2] = RelaxationSide.BOTH
        if isinstance(arg2, numeric_expr.ProductExpression):  # see Pyomo issue #1147
            arg2_arg1 = arg2.args[0]
            arg2_arg2 = arg2.args[1]
            try:
                arg2_arg1_val = pe.value(arg2_arg1)
            except ValueError:
                arg2_arg1_val = None
            try:
                arg2_arg2_val = pe.value(arg2_arg2)
            except ValueError:
                arg2_arg2_val = None
            if arg2_arg1_val == 0 or arg2_arg2_val == 0:
                arg2_val = 0
            else:
                arg2_val = pe.value(arg2)
        else:
            arg2_val = pe.value(arg2)
        if arg2_val >= 0:
            relaxation_side_map[arg1] = relaxation_side_map[node]
        else:
            if relaxation_side_map[node] == RelaxationSide.UNDER:
                relaxation_side_map[arg1] = RelaxationSide.OVER
            elif relaxation_side_map[node] == RelaxationSide.OVER:
                relaxation_side_map[arg1] = RelaxationSide.UNDER
            else:
                relaxation_side_map[arg1] = RelaxationSide.BOTH
    else:
        relaxation_side_map[arg1] = RelaxationSide.BOTH
        relaxation_side_map[arg2] = RelaxationSide.BOTH


def _relax_root_to_leaf_DivisionExpression(node, relaxation_side_map):
    arg1, arg2 = node.args
    relaxation_side_map[arg1] = RelaxationSide.BOTH
    relaxation_side_map[arg2] = RelaxationSide.BOTH


def _relax_root_to_leaf_SumExpression(node, relaxation_side_map):
    relaxation_side = relaxation_side_map[node]

    for arg in node.args:
        relaxation_side_map[arg] = relaxation_side


def _relax_root_to_leaf_NegationExpression(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side = relaxation_side_map[node]
    if relaxation_side == RelaxationSide.BOTH:
        relaxation_side_map[arg] = RelaxationSide.BOTH
    elif relaxation_side == RelaxationSide.UNDER:
        relaxation_side_map[arg] = RelaxationSide.OVER
    else:
        assert relaxation_side == RelaxationSide.OVER
        relaxation_side_map[arg] = RelaxationSide.UNDER


def _relax_root_to_leaf_PowExpression(node, relaxation_side_map):
    arg1, arg2 = node.args
    relaxation_side_map[arg1] = RelaxationSide.BOTH
    relaxation_side_map[arg2] = RelaxationSide.BOTH


def _relax_root_to_leaf_sqrt(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = relaxation_side_map[node]


def _relax_root_to_leaf_exp(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = relaxation_side_map[node]


def _relax_root_to_leaf_log(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = relaxation_side_map[node]


def _relax_root_to_leaf_log10(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = relaxation_side_map[node]


def _relax_root_to_leaf_sin(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = RelaxationSide.BOTH


def _relax_root_to_leaf_cos(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = RelaxationSide.BOTH


def _relax_root_to_leaf_arctan(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = RelaxationSide.BOTH


def _relax_root_to_leaf_tan(node, relaxation_side_map):
    arg = node.args[0]
    relaxation_side_map[arg] = RelaxationSide.BOTH


_unary_root_to_leaf_map = dict()
_unary_root_to_leaf_map['exp'] = _relax_root_to_leaf_exp
_unary_root_to_leaf_map['log'] = _relax_root_to_leaf_log
_unary_root_to_leaf_map['log10'] = _relax_root_to_leaf_log10
_unary_root_to_leaf_map['sin'] = _relax_root_to_leaf_sin
_unary_root_to_leaf_map['cos'] = _relax_root_to_leaf_cos
_unary_root_to_leaf_map['atan'] = _relax_root_to_leaf_arctan
_unary_root_to_leaf_map['sqrt'] = _relax_root_to_leaf_sqrt
_unary_root_to_leaf_map['tan'] = _relax_root_to_leaf_tan


def _relax_root_to_leaf_UnaryFunctionExpression(node, relaxation_side_map):
    if node.getname() in _unary_root_to_leaf_map:
        _unary_root_to_leaf_map[node.getname()](node, relaxation_side_map)
    else:
        raise NotImplementedError('Cannot automatically relax ' + str(node))


def _relax_root_to_leaf_GeneralExpression(node, relaxation_side_map):
    relaxation_side = relaxation_side_map[node]
    relaxation_side_map[node.expr] = relaxation_side


_relax_root_to_leaf_map = dict()
_relax_root_to_leaf_map[numeric_expr.ProductExpression] = _relax_root_to_leaf_ProductExpression
_relax_root_to_leaf_map[numeric_expr.SumExpression] = _relax_root_to_leaf_SumExpression
_relax_root_to_leaf_map[numeric_expr.LinearExpression] = _relax_root_to_leaf_SumExpression
_relax_root_to_leaf_map[numeric_expr.MonomialTermExpression] = _relax_root_to_leaf_ProductExpression
_relax_root_to_leaf_map[numeric_expr.NegationExpression] = _relax_root_to_leaf_NegationExpression
_relax_root_to_leaf_map[numeric_expr.PowExpression] = _relax_root_to_leaf_PowExpression
_relax_root_to_leaf_map[numeric_expr.DivisionExpression] = _relax_root_to_leaf_DivisionExpression
_relax_root_to_leaf_map[numeric_expr.UnaryFunctionExpression] = _relax_root_to_leaf_UnaryFunctionExpression
_relax_root_to_leaf_map[numeric_expr.NPV_ProductExpression] = _relax_root_to_leaf_ProductExpression
_relax_root_to_leaf_map[numeric_expr.NPV_SumExpression] = _relax_root_to_leaf_SumExpression
_relax_root_to_leaf_map[numeric_expr.NPV_NegationExpression] = _relax_root_to_leaf_NegationExpression
_relax_root_to_leaf_map[numeric_expr.NPV_PowExpression] = _relax_root_to_leaf_PowExpression
_relax_root_to_leaf_map[numeric_expr.NPV_DivisionExpression] = _relax_root_to_leaf_DivisionExpression
_relax_root_to_leaf_map[numeric_expr.NPV_UnaryFunctionExpression] = _relax_root_to_leaf_UnaryFunctionExpression
_relax_root_to_leaf_map[_GeneralExpressionData] = _relax_root_to_leaf_GeneralExpression
_relax_root_to_leaf_map[SimpleExpression] = _relax_root_to_leaf_GeneralExpression


class _FactorableRelaxationVisitor(ExpressionValueVisitor):
    """
    This walker generates new constraints with nonlinear terms replaced by
    auxiliary variables, and relaxations relating the auxilliary variables to
    the original variables.
    """
    def __init__(self, aux_var_map, parent_block, relaxation_side_map, counter, degree_map):
        self.aux_var_map = aux_var_map
        self.parent_block = parent_block
        self.relaxation_side_map = relaxation_side_map
        self.counter = counter
        self.degree_map = degree_map

    def visit(self, node, values):
        if node.__class__ in _relax_leaf_to_root_map:
            res = _relax_leaf_to_root_map[node.__class__](node, values, self.aux_var_map, self.degree_map,
                                                          self.parent_block, self.relaxation_side_map, self.counter)
            return res
        else:
            raise NotImplementedError('Cannot relax an expression of type ' + str(type(node)))

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.degree_map[node] = 0
            return True, node

        if node.is_variable_type():
            if node.fixed:
                self.degree_map[node] = 0
            else:
                self.degree_map[node] = 1
            return True, node

        if not node.is_expression_type():
            self.degree_map[node] = 0
            return True, node

        if node.__class__ in _relax_root_to_leaf_map:
            _relax_root_to_leaf_map[node.__class__](node, self.relaxation_side_map)
        else:
            raise NotImplementedError('Cannot relax an expression of type ' + str(type(node)))

        return False, None


def _get_prefix_notation(expr):
    pn = convert_expression_to_prefix_notation(expr, include_named_exprs=False)
    res = list()
    for i in pn:
        itype = type(i)
        if itype is tuple or itype in nonpyomo_leaf_types:
            res.append(i)
        elif isinstance(i, NumericValue):
            if i.is_fixed():
                res.append(pe.value(i))
            else:
                assert i.is_variable_type()
                res.append(id(i))
        else:
            raise NotImplementedError(f'unexpected entry in prefix notation: {str(i)}')
    return tuple(res)


def _relax_expr(expr, aux_var_map, parent_block, relaxation_side_map, counter, degree_map):
    visitor = _FactorableRelaxationVisitor(aux_var_map=aux_var_map, parent_block=parent_block,
                                           relaxation_side_map=relaxation_side_map, counter=counter,
                                           degree_map=degree_map)
    new_expr = visitor.dfs_postorder_stack(expr)
    return new_expr


def _relax_split_expr(
    expr: ExpressionBase,
    aux_var_map: MutableMapping[
        Tuple,
        Tuple[NumericValue,
              Union[BaseRelaxationData,
                    Tuple[BaseRelaxationData, BaseRelaxationData]]]
    ],
    parent_block: _BlockData,
    relaxation_side_map: MutableMapping[NumericValue, RelaxationSide],
    counter: RelaxationCounter,
    degree_map: MutableMapping[NumericValue, int],
    eigenvalue_bounder: EigenValueBounder,
    max_vars_per_alpha_bb: int,
    max_eigenvalue_for_alpha_bb: float,
    eigenvalue_opt: Optional[appsi.base.Solver],
) -> NumericValue:
    relaxation_side = relaxation_side_map[expr]
    hessian = Hessian(expr, opt=eigenvalue_opt, method=eigenvalue_bounder)
    vlist = hessian.variables()
    min_eig = hessian.get_minimum_eigenvalue()
    max_eig = hessian.get_maximum_eigenvalue()
    is_convex = min_eig >= 0
    is_concave = max_eig <= 0

    all_vars_bounded = True
    for v in vlist:
        v_lb, v_ub = v.bounds
        if v_lb is None or v_ub is None:
            all_vars_bounded = False
            break

    if len(vlist) == 1 and (is_convex or is_concave):
        pn = _get_prefix_notation(expr)
        if pn in aux_var_map:
            new_expr, relaxation = aux_var_map[pn]
            if relaxation_side != relaxation.relaxation_side:
                relaxation.relaxation_side = RelaxationSide.BOTH
        else:
            new_expr = _get_aux_var(parent_block, expr)
            relaxation = PWUnivariateRelaxation()
            if is_convex:
                shape = FunctionShape.CONVEX
            else:
                shape = FunctionShape.CONCAVE
            relaxation.set_input(
                x=vlist[0], aux_var=new_expr, relaxation_side=relaxation_side,
                f_x_expr=expr, shape=shape,
            )
            aux_var_map[pn] = (new_expr, relaxation)
            setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
            counter.increment()
            degree_map[new_expr] = 1
    elif ((is_convex and relaxation_side == RelaxationSide.UNDER)
          or (is_concave and relaxation_side == RelaxationSide.OVER)):
        pn = _get_prefix_notation(expr)
        if pn in aux_var_map:
            new_expr, (underestimator, overestimator) = aux_var_map[pn]
        else:
            new_expr, underestimator, overestimator = None, None, None
        if new_expr is None:
            new_expr = _get_aux_var(parent_block, expr)
        if (
            (is_convex and underestimator is None)
            or (is_concave and overestimator is None)
        ):
            relaxation = MultivariateRelaxation()
            if is_convex:
                shape = FunctionShape.CONVEX
                underestimator = relaxation
            else:
                shape = FunctionShape.CONCAVE
                overestimator = relaxation
            relaxation.set_input(
                aux_var=new_expr, shape=shape, f_x_expr=expr,
            )
            aux_var_map[pn] = (new_expr, (underestimator, overestimator))
            setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
            counter.increment()
            degree_map[new_expr] = 1
    elif (
        all_vars_bounded
        and len(vlist) <= max_vars_per_alpha_bb
        and (
            (relaxation_side == RelaxationSide.UNDER and min_eig >= -abs(max_eigenvalue_for_alpha_bb))
            or (relaxation_side == RelaxationSide.OVER and max_eig <= abs(max_eigenvalue_for_alpha_bb))
        )
    ):
        pn = _get_prefix_notation(expr)
        if pn in aux_var_map:
            new_expr, (underestimator, overestimator) = aux_var_map[pn]
        else:
            new_expr, underestimator, overestimator = None, None, None
        if new_expr is None:
            new_expr = _get_aux_var(parent_block, expr)
        if (
            (relaxation_side == RelaxationSide.UNDER and underestimator is None)
            or (relaxation_side == RelaxationSide.OVER and overestimator is None)
        ):
            relaxation = AlphaBBRelaxation()
            relaxation.set_input(
                aux_var=new_expr,
                f_x_expr=expr,
                relaxation_side=relaxation_side,
                hessian=hessian,
            )
            if relaxation_side == RelaxationSide.UNDER:
                underestimator = relaxation
            else:
                overestimator = relaxation
            aux_var_map[pn] = (new_expr, (underestimator, overestimator))
            setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
            counter.increment()
            degree_map[new_expr] = 1
    else:
        visitor = _FactorableRelaxationVisitor(aux_var_map=aux_var_map,
                                               parent_block=parent_block,
                                               relaxation_side_map=relaxation_side_map,
                                               counter=counter,
                                               degree_map=degree_map)
        new_expr = visitor.dfs_postorder_stack(expr)
    return new_expr


def _relax_expr_with_convexity_check(
    orig_expr: ExpressionBase,
    aux_var_map: MutableMapping[
        Tuple,
        Tuple[NumericValue,
              Union[BaseRelaxationData,
                    Tuple[BaseRelaxationData, BaseRelaxationData]]]
    ],
    parent_block: _BlockData,
    relaxation_side_map: MutableMapping[NumericValue, RelaxationSide],
    counter: RelaxationCounter,
    degree_map: MutableMapping[NumericValue, int],
    perform_expression_simplification: bool,
    eigenvalue_bounder: EigenValueBounder,
    max_vars_per_alpha_bb: int,
    max_eigenvalue_for_alpha_bb: float,
    eigenvalue_opt: Optional[appsi.base.Solver],
):
    if relaxation_side_map[orig_expr] == RelaxationSide.BOTH:
        res_list = []
        for side in [RelaxationSide.UNDER, RelaxationSide.OVER]:
            relaxation_side_map[orig_expr] = side
            tmp_res = _relax_expr_with_convexity_check(
                orig_expr=orig_expr,
                aux_var_map=aux_var_map,
                parent_block=parent_block,
                relaxation_side_map=relaxation_side_map,
                counter=counter,
                degree_map=degree_map,
                perform_expression_simplification=perform_expression_simplification,
                eigenvalue_bounder=eigenvalue_bounder,
                max_vars_per_alpha_bb=max_vars_per_alpha_bb,
                max_eigenvalue_for_alpha_bb=max_eigenvalue_for_alpha_bb,
                eigenvalue_opt=eigenvalue_opt,
            )
            res_list.append(tmp_res)
        linking_expr = res_list[0] - res_list[1]
        linking_repn = generate_standard_repn(linking_expr, compute_values=False, quadratic=True)
        linking_expr = linking_repn.to_expression()
        if is_constant(linking_expr):
            assert value(linking_expr) == 0
        else:
            parent_block.aux_cons.add(linking_repn.to_expression() == 0)
        res = res_list[0]
        relaxation_side_map[orig_expr] = RelaxationSide.BOTH
    else:
        if perform_expression_simplification:
            _expr = simplify_expr(orig_expr)
        else:
            _expr = orig_expr
        list_of_exprs = split_expr(_expr)
        list_of_new_exprs = list()

        for expr in list_of_exprs:
            relaxation_side_map[expr] = relaxation_side_map[orig_expr]
            new_expr = _relax_split_expr(
                expr=expr,
                aux_var_map=aux_var_map,
                parent_block=parent_block,
                relaxation_side_map=relaxation_side_map,
                counter=counter,
                degree_map=degree_map,
                eigenvalue_bounder=eigenvalue_bounder,
                max_vars_per_alpha_bb=max_vars_per_alpha_bb,
                max_eigenvalue_for_alpha_bb=max_eigenvalue_for_alpha_bb,
                eigenvalue_opt=eigenvalue_opt,
            )
            list_of_new_exprs.append(new_expr)
        res = sum(list_of_new_exprs)
    return res


def relax(
    model,
    descend_into=None,
    in_place=False,
    use_fbbt=True,
    fbbt_options=None,
    perform_expression_simplification: bool = True,
    use_alpha_bb: bool = False,
    eigenvalue_bounder: EigenValueBounder = EigenValueBounder.GershgorinWithSimplification,
    max_vars_per_alpha_bb: int = 4,
    max_eigenvalue_for_alpha_bb: float = 100,
    eigenvalue_opt: Optional[appsi.base.Solver] = None,
):
    """
    Create a convex relaxation of the model.

    Parameters
    ----------
    model: pyomo.core.base.block._BlockData or pyomo.core.base.PyomoModel.ConcreteModel
        The model or block to be relaxed
    descend_into: type or tuple of type, optional
        The types of pyomo components that should be checked for constraints to be relaxed. The
        default is (Block, Disjunct).
    in_place: bool, optional
        If False (default=False), model will be cloned, and the clone will be relaxed. 
        If True, then model will be modified in place.
    use_fbbt: bool, optional
        If True (default=True), then FBBT will be used to tighten variable bounds. If False, 
        FBBT will not be used.
    fbbt_options: dict, optional
        The options to pass to the call to fbbt. See pyomo.contrib.fbbt.fbbt.fbbt for details.
    convexity_effort: ConvexityEffort

    Returns
    -------
    m: pyomo.core.base.block._BlockData or pyomo.core.base.PyomoModel.ConcreteModel
        The relaxed model
    """
    """
    For now, we will use FBBT both before relaxing the model and after relaxing the model. The reason we need to 
    do it before relaxing the model is that the variable bounds will affect the structure of the relaxation. For 
    example, if we need to relax x**3 and x >= 0, then we know x**3 is convex, and we can relax it as a 
    convex, univariate function. However, if x can be positive or negative, then x**3 is neither convex nor concave.
    In this case, we relax it by reformulating it as x * x**2. The hope is that performing FBBT before relaxing 
    the model will help identify things like x >= 0 and therefore x**3 is convex. The correct way to do this is to 
    update the relaxation classes so that the original expression is known, and the best relaxation can be used 
    anytime the variable bounds are updated. For example, suppose the model is relaxed and, only after OBBT is 
    performed, we find out x >= 0. We should be able to easily update the relaxation so that x**3 is then relaxed 
    as a convex univariate function. The reason FBBT needs to be performed after relaxing the model is that 
    we want to make sure that all of the auxilliary variables introduced get tightened bounds. The correct way to 
    handle this is to perform FBBT with the original model with suspect, which forms a DAG. Each auxilliary variable 
    introduced in the relaxed model corresponds to a node in the DAG. If we use suspect, then we can easily 
    update the bounds of the auxilliary variables without performing FBBT a second time.
    """
    if not in_place:
        m = model.clone()
    else:
        m = model

    if fbbt_options is None:
        fbbt_options = dict()

    if use_fbbt:
        it = appsi.fbbt.IntervalTightener()
        for k, v in fbbt_options.items():
            setattr(it.config, k, v)
        original_active_vars = ComponentSet(active_vars(m, include_fixed=False))
        it.perform_fbbt(m)
        new_active_vars = ComponentSet(active_vars(m, include_fixed=False))
        # some variables may have become stale by deactivating satisfied constraints,
        # so we need to fix them.
        for v in original_active_vars - new_active_vars:
            v.fix(0.5 * (v.lb + v.ub))

    if descend_into is None:
        descend_into = (pe.Block, Disjunct)

    aux_var_map = dict()
    counter_dict = dict()
    degree_map = ComponentMap()

    for c in nonrelaxation_component_data_objects(m, ctype=Constraint, active=True, descend_into=descend_into, sort=True):
        body_degree = polynomial_degree(c.body)
        if body_degree is not None:
            if body_degree <= 1:
                continue

        if c.lower is not None and c.upper is not None:
            relaxation_side = RelaxationSide.BOTH
        elif c.lower is not None:
            relaxation_side = RelaxationSide.OVER
        elif c.upper is not None:
            relaxation_side = RelaxationSide.UNDER
        else:
            raise ValueError('Encountered a constraint without a lower or an upper bound: ' + str(c))

        parent_block = c.parent_block()

        if parent_block in counter_dict:
            counter = counter_dict[parent_block]
        else:
            parent_block.relaxations = pe.Block()
            parent_block.aux_vars = pe.VarList()
            parent_block.aux_cons = pe.ConstraintList()
            counter = RelaxationCounter()
            counter_dict[parent_block] = counter

        repn = generate_standard_repn(c.body, quadratic=False, compute_values=False)
        assert len(repn.quadratic_vars) == 0
        assert repn.nonlinear_expr is not None
        if len(repn.linear_vars) > 0:
            new_body = numeric_expr.LinearExpression(constant=repn.constant, linear_coefs=repn.linear_coefs, linear_vars=repn.linear_vars)
        else:
            new_body = repn.constant

        relaxation_side_map = ComponentMap()
        relaxation_side_map[repn.nonlinear_expr] = relaxation_side

        if not use_alpha_bb:
            new_body += _relax_expr(
                expr=repn.nonlinear_expr, aux_var_map=aux_var_map,
                parent_block=parent_block, relaxation_side_map=relaxation_side_map,
                counter=counter, degree_map=degree_map
            )
        else:
            new_body += _relax_expr_with_convexity_check(
                orig_expr=repn.nonlinear_expr, aux_var_map=aux_var_map,
                parent_block=parent_block, relaxation_side_map=relaxation_side_map,
                counter=counter, degree_map=degree_map,
                perform_expression_simplification=perform_expression_simplification,
                eigenvalue_bounder=eigenvalue_bounder,
                max_vars_per_alpha_bb=max_vars_per_alpha_bb,
                max_eigenvalue_for_alpha_bb=max_eigenvalue_for_alpha_bb,
                eigenvalue_opt=eigenvalue_opt,
            )
        lb = c.lower
        ub = c.upper
        parent_block.aux_cons.add(pe.inequality(lb, new_body, ub))
        parent_component = c.parent_component()
        if parent_component.is_indexed():
            del parent_component[c.index()]
        else:
            parent_block.del_component(c)

    for c in nonrelaxation_component_data_objects(m, ctype=pe.Objective, active=True, descend_into=descend_into, sort=True):
        degree = polynomial_degree(c.expr)
        if degree is not None:
            if degree <= 1:
                continue

        if c.sense == pe.minimize:
            relaxation_side = RelaxationSide.UNDER
        elif c.sense == pe.maximize:
            relaxation_side = RelaxationSide.OVER
        else:
            raise ValueError('Encountered an objective with an unrecognized sense: ' + str(c))

        parent_block = c.parent_block()

        if parent_block in counter_dict:
            counter = counter_dict[parent_block]
        else:
            parent_block.relaxations = pe.Block()
            parent_block.aux_vars = pe.VarList()
            parent_block.aux_cons = pe.ConstraintList()
            counter = RelaxationCounter()
            counter_dict[parent_block] = counter

        if not hasattr(parent_block, 'aux_objectives'):
            parent_block.aux_objectives = pe.ObjectiveList()

        repn = generate_standard_repn(c.expr, quadratic=False, compute_values=False)
        assert len(repn.quadratic_vars) == 0
        assert repn.nonlinear_expr is not None
        if len(repn.linear_vars) > 0:
            new_body = numeric_expr.LinearExpression(constant=repn.constant, linear_coefs=repn.linear_coefs, linear_vars=repn.linear_vars)
        else:
            new_body = repn.constant

        relaxation_side_map = ComponentMap()
        relaxation_side_map[repn.nonlinear_expr] = relaxation_side

        if not use_alpha_bb:
            new_body += _relax_expr(
                expr=repn.nonlinear_expr, aux_var_map=aux_var_map,
                parent_block=parent_block, relaxation_side_map=relaxation_side_map,
                counter=counter, degree_map=degree_map
            )
        else:
            new_body += _relax_expr_with_convexity_check(
                orig_expr=repn.nonlinear_expr, aux_var_map=aux_var_map,
                parent_block=parent_block, relaxation_side_map=relaxation_side_map,
                counter=counter, degree_map=degree_map,
                perform_expression_simplification=perform_expression_simplification,
                eigenvalue_bounder=eigenvalue_bounder,
                max_vars_per_alpha_bb=max_vars_per_alpha_bb,
                max_eigenvalue_for_alpha_bb=max_eigenvalue_for_alpha_bb,
                eigenvalue_opt=eigenvalue_opt,
            )
        sense = c.sense
        parent_block.aux_objectives.add(new_body, sense=sense)
        parent_component = c.parent_component()
        if parent_component.is_indexed():
            del parent_component[c.index()]
        else:
            parent_block.del_component(c)

    if use_fbbt:
        for relaxation in relaxation_data_objects(m, descend_into=True, active=True):
            relaxation.rebuild(build_nonlinear_constraint=True)

        it = appsi.fbbt.IntervalTightener()
        for k, v in fbbt_options.items():
            setattr(it.config, k, v)
        it.config.deactivate_satisfied_constraints = False
        it.perform_fbbt(m)

        for relaxation in relaxation_data_objects(m, descend_into=True, active=True):
            relaxation.use_linear_relaxation = True
            relaxation.rebuild()
    else:
        for relaxation in relaxation_data_objects(m, descend_into=True, active=True):
            relaxation.use_linear_relaxation = True
            relaxation.rebuild()

    return m
