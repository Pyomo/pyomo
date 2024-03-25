import pyomo.environ as pe
from pyomo.common.collections import ComponentMap
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import ExpressionValueVisitor
from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
    NumericValue,
    is_fixed,
    is_constant,
)
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr, fbbt
import math
import logging
from .univariate import (
    PWUnivariateRelaxation,
    PWXSquaredRelaxation,
    PWCosRelaxation,
    PWSinRelaxation,
    PWArctanRelaxation,
)
from .mccormick import PWMcCormickRelaxation
from pyomo.contrib.coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from pyomo.core.base.expression import _GeneralExpressionData, SimpleExpression
from pyomo.repn.standard_repn import generate_standard_repn
from .iterators import relaxation_data_objects
from pyomo.contrib.coramin.clone import clone_shallow_active_flat


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
                raise NotImplementedError(f'unexpected entry: {str(i)}')
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
        parent_block.vars.append(_var)
        _con = parent_block.linear.cons.add(_var == arg)
        fbbt(_con)
        return _var
    else:
        return arg


def _get_aux_var(parent_block, expr):
    _aux_var = parent_block.aux_vars.add()
    parent_block.vars.append(_aux_var)
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


def _relax_leaf_to_root_ProductExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        elif (
            arg1.__class__ == numeric_expr.MonomialTermExpression
            or arg2.__class__ == numeric_expr.MonomialTermExpression
        ):
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
            res = _relax_expr(
                expr=_reformulated,
                aux_var_map=aux_var_map,
                parent_block=parent_block,
                relaxation_side_map=_new_relaxation_side_map,
                counter=counter,
                degree_map=degree_map,
            )
            degree_map[res] = 1
        elif arg1 is arg2:
            # reformulate arg1 * arg2 as arg1**2
            _new_relaxation_side_map = ComponentMap()
            _reformulated = arg1**2
            _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
            res = _relax_expr(
                expr=_reformulated,
                aux_var_map=aux_var_map,
                parent_block=parent_block,
                relaxation_side_map=_new_relaxation_side_map,
                counter=counter,
                degree_map=degree_map,
            )
            degree_map[res] = 1
        else:
            _aux_var = _get_aux_var(parent_block, arg1 * arg2)
            arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
            arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
            relaxation_side = relaxation_side_map[node]
            relaxation = PWMcCormickRelaxation()
            relaxation.set_input(
                x1=arg1, x2=arg2, aux_var=_aux_var, relaxation_side=relaxation_side
            )
            aux_var_map[h1] = (_aux_var, relaxation)
            setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
            counter.increment()
            res = _aux_var
            degree_map[res] = 1
    return res


def _relax_leaf_to_root_DivisionExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
    coef = coef1 / coef2
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
            _aux_var = _get_aux_var(parent_block, 1 / arg2)
            arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
            relaxation_side = relaxation_side_map[node]
            degree_map[_aux_var] = 1
            if compute_float_bounds_on_expr(arg2)[0] > 0:
                relaxation = PWUnivariateRelaxation()
                relaxation.set_input(
                    x=arg2,
                    aux_var=_aux_var,
                    relaxation_side=relaxation_side,
                    f_x_expr=1 / arg2,
                    shape=FunctionShape.CONVEX,
                )
            elif compute_float_bounds_on_expr(arg2)[1] < 0:
                relaxation = PWUnivariateRelaxation()
                relaxation.set_input(
                    x=arg2,
                    aux_var=_aux_var,
                    relaxation_side=relaxation_side,
                    f_x_expr=1 / arg2,
                    shape=FunctionShape.CONCAVE,
                )
            else:
                _one = parent_block.aux_vars.add()
                _one.fix(1)
                relaxation = PWMcCormickRelaxation()
                relaxation.set_input(
                    x1=arg2, x2=_aux_var, aux_var=_one, relaxation_side=relaxation_side
                )
            aux_var_map[h2] = (_aux_var, relaxation)
            setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
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
        relaxation.set_input(
            x1=arg2, x2=_aux_var, aux_var=arg1, relaxation_side=relaxation_side
        )
        aux_var_map[h1] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        res = coef * _aux_var
        degree_map[_aux_var] = 1
        degree_map[res] = 1
        return res


def _relax_quadratic(
    arg1, aux_var_map, relaxation_side, degree_map, parent_block, counter
):
    _aux_var = _get_aux_var(parent_block, arg1**2)
    arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
    degree_map[_aux_var] = 1
    relaxation = PWXSquaredRelaxation()
    relaxation.set_input(x=arg1, aux_var=_aux_var, relaxation_side=relaxation_side)
    aux_var_map[Hashable(arg1, 2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_convex_pow(
    arg1,
    arg2,
    aux_var_map,
    relaxation_side,
    degree_map,
    parent_block,
    counter,
    swap=False,
):
    _aux_var = _get_aux_var(parent_block, arg1**arg2)
    if swap:
        arg2 = replace_sub_expression_with_aux_var(arg2, parent_block)
        _x = arg2
    else:
        arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
        _x = arg1
        assert type(arg2) in {int, float}
        if round(arg2) != arg2:
            if arg1.lb is None or arg1.lb < 0:
                arg1.setlb(0)
    degree_map[_aux_var] = 1
    relaxation = PWUnivariateRelaxation()
    relaxation.set_input(
        x=_x,
        aux_var=_aux_var,
        relaxation_side=relaxation_side,
        f_x_expr=arg1**arg2,
        shape=FunctionShape.CONVEX,
    )
    aux_var_map[Hashable(arg1, arg2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_concave_pow(
    arg1, arg2, aux_var_map, relaxation_side, degree_map, parent_block, counter
):
    _aux_var = _get_aux_var(parent_block, arg1**arg2)
    arg1 = replace_sub_expression_with_aux_var(arg1, parent_block)
    degree_map[_aux_var] = 1
    relaxation = PWUnivariateRelaxation()
    relaxation.set_input(
        x=arg1,
        aux_var=_aux_var,
        relaxation_side=relaxation_side,
        f_x_expr=arg1**arg2,
        shape=FunctionShape.CONCAVE,
    )
    aux_var_map[Hashable(arg1, arg2, 'pow')] = (_aux_var, relaxation)
    setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
    counter.increment()
    return _aux_var


def _relax_leaf_to_root_PowExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
                res = arg1**arg2
                degree_map[res] = 0
                return res
            if not is_constant(arg2):
                logger.debug(
                    'Only constant exponents are supported: '
                    + str(arg1**arg2)
                    + '\nReplacing '
                    + str(arg2)
                    + ' with its value.'
                )
            arg2 = pe.value(arg2)
            if arg2 == 1:
                return arg1
            elif arg2 == 0:
                res = 1
                degree_map[res] = 0
                return res
            elif arg2 == 2:
                return _relax_quadratic(
                    arg1=arg1,
                    aux_var_map=aux_var_map,
                    relaxation_side=relaxation_side_map[node],
                    degree_map=degree_map,
                    parent_block=parent_block,
                    counter=counter,
                )
            elif arg2 >= 0:
                if arg2 == round(arg2):
                    if arg2 % 2 == 0 or compute_float_bounds_on_expr(arg1)[0] >= 0:
                        return _relax_convex_pow(
                            arg1=arg1,
                            arg2=arg2,
                            aux_var_map=aux_var_map,
                            relaxation_side=relaxation_side_map[node],
                            degree_map=degree_map,
                            parent_block=parent_block,
                            counter=counter,
                        )
                    elif compute_float_bounds_on_expr(arg1)[1] <= 0:
                        return _relax_concave_pow(
                            arg1=arg1,
                            arg2=arg2,
                            aux_var_map=aux_var_map,
                            relaxation_side=relaxation_side_map[node],
                            degree_map=degree_map,
                            parent_block=parent_block,
                            counter=counter,
                        )
                    else:  # reformulate arg1 ** arg2 as arg1 * arg1 ** (arg2 - 1)
                        _new_relaxation_side_map = ComponentMap()
                        _reformulated = arg1 * arg1 ** (arg2 - 1)
                        _new_relaxation_side_map[_reformulated] = relaxation_side_map[
                            node
                        ]
                        res = _relax_expr(
                            expr=_reformulated,
                            aux_var_map=aux_var_map,
                            parent_block=parent_block,
                            relaxation_side_map=_new_relaxation_side_map,
                            counter=counter,
                            degree_map=degree_map,
                        )
                        degree_map[res] = 1
                        return res
                else:
                    if arg2 < 1:
                        return _relax_concave_pow(
                            arg1=arg1,
                            arg2=arg2,
                            aux_var_map=aux_var_map,
                            relaxation_side=relaxation_side_map[node],
                            degree_map=degree_map,
                            parent_block=parent_block,
                            counter=counter,
                        )
                    else:
                        return _relax_convex_pow(
                            arg1=arg1,
                            arg2=arg2,
                            aux_var_map=aux_var_map,
                            relaxation_side=relaxation_side_map[node],
                            degree_map=degree_map,
                            parent_block=parent_block,
                            counter=counter,
                        )
            else:
                if arg2 == round(arg2):
                    if compute_float_bounds_on_expr(arg1)[0] >= 0:
                        return _relax_convex_pow(
                            arg1=arg1,
                            arg2=arg2,
                            aux_var_map=aux_var_map,
                            relaxation_side=relaxation_side_map[node],
                            degree_map=degree_map,
                            parent_block=parent_block,
                            counter=counter,
                        )
                    elif compute_float_bounds_on_expr(arg1)[1] <= 0:
                        if arg2 % 2 == 0:
                            return _relax_convex_pow(
                                arg1=arg1,
                                arg2=arg2,
                                aux_var_map=aux_var_map,
                                relaxation_side=relaxation_side_map[node],
                                degree_map=degree_map,
                                parent_block=parent_block,
                                counter=counter,
                            )
                        else:
                            return _relax_concave_pow(
                                arg1=arg1,
                                arg2=arg2,
                                aux_var_map=aux_var_map,
                                relaxation_side=relaxation_side_map[node],
                                degree_map=degree_map,
                                parent_block=parent_block,
                                counter=counter,
                            )
                    else:
                        # reformulate arg1 ** arg2 as 1 / arg1 ** (-arg2)
                        _new_relaxation_side_map = ComponentMap()
                        _reformulated = 1 / (arg1 ** (-arg2))
                        _new_relaxation_side_map[_reformulated] = relaxation_side_map[
                            node
                        ]
                        res = _relax_expr(
                            expr=_reformulated,
                            aux_var_map=aux_var_map,
                            parent_block=parent_block,
                            relaxation_side_map=_new_relaxation_side_map,
                            counter=counter,
                            degree_map=degree_map,
                        )
                        degree_map[res] = 1
                        return res
                else:
                    return _relax_convex_pow(
                        arg1=arg1,
                        arg2=arg2,
                        aux_var_map=aux_var_map,
                        relaxation_side=relaxation_side_map[node],
                        degree_map=degree_map,
                        parent_block=parent_block,
                        counter=counter,
                    )
        elif degree1 == 0:
            if not is_constant(arg1):
                logger.debug(
                    'Found {0} raised to a variable power. However, {0} does not appear to be constant (maybe '
                    'it is or depends on a mutable Param?). Replacing {0} with its value.'.format(
                        str(arg1)
                    )
                )
                arg1 = pe.value(arg1)
            if arg1 < 0:
                raise ValueError(
                    'Cannot raise a negative base to a variable exponent: '
                    + str(arg1**arg2)
                )
            return _relax_convex_pow(
                arg1=arg1,
                arg2=arg2,
                aux_var_map=aux_var_map,
                relaxation_side=relaxation_side_map[node],
                degree_map=degree_map,
                parent_block=parent_block,
                counter=counter,
                swap=True,
            )
        else:
            assert compute_float_bounds_on_expr(arg1)[0] >= 0
            _new_relaxation_side_map = ComponentMap()
            _reformulated = pe.exp(arg2 * pe.log(arg1))
            _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
            res = _relax_expr(
                expr=_reformulated,
                aux_var_map=aux_var_map,
                parent_block=parent_block,
                relaxation_side_map=_new_relaxation_side_map,
                counter=counter,
                degree_map=degree_map,
            )
            degree_map[res] = 1
            return res


def _relax_leaf_to_root_SumExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
    res = sum(values)
    degree_map[res] = max([degree_map[arg] for arg in values])
    return res


def _relax_leaf_to_root_NegationExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
    arg = values[0]
    res = -arg
    degree_map[res] = degree_map[arg]
    return res


def _relax_leaf_to_root_sqrt(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
    arg = values[0]
    _new_relaxation_side_map = ComponentMap()
    _reformulated = arg**0.5
    _new_relaxation_side_map[_reformulated] = relaxation_side_map[node]
    res = _relax_expr(
        expr=_reformulated,
        aux_var_map=aux_var_map,
        parent_block=parent_block,
        relaxation_side_map=_new_relaxation_side_map,
        counter=counter,
        degree_map=degree_map,
    )
    degree_map[res] = 1
    return res


def _relax_leaf_to_root_exp(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        relaxation.set_input(
            x=arg,
            aux_var=_aux_var,
            relaxation_side=relaxation_side,
            f_x_expr=pe.exp(arg),
            shape=FunctionShape.CONVEX,
        )
        aux_var_map[id(arg), 'exp'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_log(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        relaxation.set_input(
            x=arg,
            aux_var=_aux_var,
            relaxation_side=relaxation_side,
            f_x_expr=pe.log(arg),
            shape=FunctionShape.CONCAVE,
        )
        aux_var_map[id(arg), 'log'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_log10(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        relaxation.set_input(
            x=arg,
            aux_var=_aux_var,
            relaxation_side=relaxation_side,
            f_x_expr=pe.log10(arg),
            shape=FunctionShape.CONCAVE,
        )
        aux_var_map[id(arg), 'log10'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_sin(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_cos(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_arctan(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
        counter.increment()
        return _aux_var


def _relax_leaf_to_root_tan(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
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

        if arg.lb >= 0 and arg.ub <= math.pi / 2:
            relaxation = PWUnivariateRelaxation()
            relaxation.set_input(
                x=arg,
                aux_var=_aux_var,
                shape=FunctionShape.CONVEX,
                f_x_expr=pe.tan(arg),
                relaxation_side=relaxation_side,
            )
        elif arg.lb >= -math.pi / 2 and arg.ub <= 0:
            relaxation = PWUnivariateRelaxation()
            relaxation.set_input(
                x=arg,
                aux_var=_aux_var,
                shape=FunctionShape.CONCAVE,
                f_x_expr=pe.tan(arg),
                relaxation_side=relaxation_side,
            )
        else:
            raise NotImplementedError('Use alpha-BB here')
        aux_var_map[id(arg), 'tan'] = (_aux_var, relaxation)
        setattr(parent_block.relaxations, 'rel' + str(counter), relaxation)
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


def _relax_leaf_to_root_UnaryFunctionExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
    if node.getname() in _unary_leaf_to_root_map:
        return _unary_leaf_to_root_map[node.getname()](
            node=node,
            values=values,
            aux_var_map=aux_var_map,
            degree_map=degree_map,
            parent_block=parent_block,
            relaxation_side_map=relaxation_side_map,
            counter=counter,
        )
    else:
        raise NotImplementedError('Cannot automatically relax ' + str(node))


def _relax_leaf_to_root_GeneralExpression(
    node, values, aux_var_map, degree_map, parent_block, relaxation_side_map, counter
):
    arg = values[0]
    return arg


_relax_leaf_to_root_map = dict()
_relax_leaf_to_root_map[numeric_expr.ProductExpression] = (
    _relax_leaf_to_root_ProductExpression
)
_relax_leaf_to_root_map[numeric_expr.SumExpression] = _relax_leaf_to_root_SumExpression
_relax_leaf_to_root_map[numeric_expr.LinearExpression] = (
    _relax_leaf_to_root_SumExpression
)
_relax_leaf_to_root_map[numeric_expr.MonomialTermExpression] = (
    _relax_leaf_to_root_ProductExpression
)
_relax_leaf_to_root_map[numeric_expr.NegationExpression] = (
    _relax_leaf_to_root_NegationExpression
)
_relax_leaf_to_root_map[numeric_expr.PowExpression] = _relax_leaf_to_root_PowExpression
_relax_leaf_to_root_map[numeric_expr.DivisionExpression] = (
    _relax_leaf_to_root_DivisionExpression
)
_relax_leaf_to_root_map[numeric_expr.UnaryFunctionExpression] = (
    _relax_leaf_to_root_UnaryFunctionExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_ProductExpression] = (
    _relax_leaf_to_root_ProductExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_SumExpression] = (
    _relax_leaf_to_root_SumExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_NegationExpression] = (
    _relax_leaf_to_root_NegationExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_PowExpression] = (
    _relax_leaf_to_root_PowExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_DivisionExpression] = (
    _relax_leaf_to_root_DivisionExpression
)
_relax_leaf_to_root_map[numeric_expr.NPV_UnaryFunctionExpression] = (
    _relax_leaf_to_root_UnaryFunctionExpression
)
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
_relax_root_to_leaf_map[numeric_expr.ProductExpression] = (
    _relax_root_to_leaf_ProductExpression
)
_relax_root_to_leaf_map[numeric_expr.SumExpression] = _relax_root_to_leaf_SumExpression
_relax_root_to_leaf_map[numeric_expr.LinearExpression] = (
    _relax_root_to_leaf_SumExpression
)
_relax_root_to_leaf_map[numeric_expr.MonomialTermExpression] = (
    _relax_root_to_leaf_ProductExpression
)
_relax_root_to_leaf_map[numeric_expr.NegationExpression] = (
    _relax_root_to_leaf_NegationExpression
)
_relax_root_to_leaf_map[numeric_expr.PowExpression] = _relax_root_to_leaf_PowExpression
_relax_root_to_leaf_map[numeric_expr.DivisionExpression] = (
    _relax_root_to_leaf_DivisionExpression
)
_relax_root_to_leaf_map[numeric_expr.UnaryFunctionExpression] = (
    _relax_root_to_leaf_UnaryFunctionExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_ProductExpression] = (
    _relax_root_to_leaf_ProductExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_SumExpression] = (
    _relax_root_to_leaf_SumExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_NegationExpression] = (
    _relax_root_to_leaf_NegationExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_PowExpression] = (
    _relax_root_to_leaf_PowExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_DivisionExpression] = (
    _relax_root_to_leaf_DivisionExpression
)
_relax_root_to_leaf_map[numeric_expr.NPV_UnaryFunctionExpression] = (
    _relax_root_to_leaf_UnaryFunctionExpression
)
_relax_root_to_leaf_map[_GeneralExpressionData] = _relax_root_to_leaf_GeneralExpression
_relax_root_to_leaf_map[SimpleExpression] = _relax_root_to_leaf_GeneralExpression


class _FactorableRelaxationVisitor(ExpressionValueVisitor):
    """
    This walker generates new constraints with nonlinear terms replaced by
    auxiliary variables, and relaxations relating the auxiliary variables to
    the original variables.
    """

    def __init__(
        self, aux_var_map, parent_block, relaxation_side_map, counter, degree_map
    ):
        self.aux_var_map = aux_var_map
        self.parent_block = parent_block
        self.relaxation_side_map = relaxation_side_map
        self.counter = counter
        self.degree_map = degree_map

    def visit(self, node, values):
        if node.__class__ in _relax_leaf_to_root_map:
            res = _relax_leaf_to_root_map[node.__class__](
                node,
                values,
                self.aux_var_map,
                self.degree_map,
                self.parent_block,
                self.relaxation_side_map,
                self.counter,
            )
            return res
        else:
            raise NotImplementedError(
                'Cannot relax an expression of type ' + str(type(node))
            )

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
            raise NotImplementedError(
                'Cannot relax an expression of type ' + str(type(node))
            )

        return False, None


def _relax_expr(
    expr, aux_var_map, parent_block, relaxation_side_map, counter, degree_map
):
    visitor = _FactorableRelaxationVisitor(
        aux_var_map=aux_var_map,
        parent_block=parent_block,
        relaxation_side_map=relaxation_side_map,
        counter=counter,
        degree_map=degree_map,
    )
    new_expr = visitor.dfs_postorder_stack(expr)
    return new_expr


def _relax_cloned_model(m):
    """
    Create a convex relaxation of the model.

    Parameters
    ----------
    m: pyomo.core.base.block._BlockData or pyomo.core.base.PyomoModel.ConcreteModel
        The model or block to be relaxed
    """
    if not hasattr(m, 'aux_vars'):
        m.aux_vars = pe.VarList()
    m.relaxations = pe.Block()

    aux_var_map = dict()
    degree_map = ComponentMap()
    counter = RelaxationCounter()

    for c in m.nonlinear.cons.values():
        repn = generate_standard_repn(c.body, quadratic=False, compute_values=True)
        assert len(repn.quadratic_vars) == 0
        if repn.nonlinear_expr is None:
            continue

        cl, cu = c.lb, c.ub
        if cl is not None and cu is not None:
            relaxation_side = RelaxationSide.BOTH
        elif cl is not None:
            relaxation_side = RelaxationSide.OVER
        elif cu is not None:
            relaxation_side = RelaxationSide.UNDER
        else:
            raise ValueError(
                'Encountered a constraint without a lower or an upper bound: ' + str(c)
            )

        if len(repn.linear_vars) > 0:
            new_body = numeric_expr.LinearExpression(
                constant=repn.constant,
                linear_coefs=repn.linear_coefs,
                linear_vars=repn.linear_vars,
            )
        else:
            new_body = repn.constant

        relaxation_side_map = ComponentMap()
        relaxation_side_map[repn.nonlinear_expr] = relaxation_side

        new_body += _relax_expr(
            expr=repn.nonlinear_expr,
            aux_var_map=aux_var_map,
            parent_block=m,
            relaxation_side_map=relaxation_side_map,
            counter=counter,
            degree_map=degree_map,
        )
        m.linear.cons.add((cl, new_body, cu))

    if hasattr(m.nonlinear, 'obj'):
        obj = m.nonlinear.obj
        if obj.sense == pe.minimize:
            relaxation_side = RelaxationSide.UNDER
        elif obj.sense == pe.maximize:
            relaxation_side = RelaxationSide.OVER
        else:
            raise ValueError(
                'Encountered an objective with an unrecognized sense: ' + str(obj)
            )

        repn = generate_standard_repn(obj.expr, quadratic=False, compute_values=True)
        assert len(repn.quadratic_vars) == 0
        assert repn.nonlinear_expr is not None
        if len(repn.linear_vars) > 0:
            new_body = numeric_expr.LinearExpression(
                constant=repn.constant,
                linear_coefs=repn.linear_coefs,
                linear_vars=repn.linear_vars,
            )
        else:
            new_body = repn.constant

        relaxation_side_map = ComponentMap()
        relaxation_side_map[repn.nonlinear_expr] = relaxation_side

        new_body += _relax_expr(
            expr=repn.nonlinear_expr,
            aux_var_map=aux_var_map,
            parent_block=m,
            relaxation_side_map=relaxation_side_map,
            counter=counter,
            degree_map=degree_map,
        )
        m.linear.obj = pe.Objective(expr=new_body, sense=obj.sense)

    del m.nonlinear

    for relaxation in relaxation_data_objects(m, descend_into=True, active=True):
        relaxation.rebuild()


def relax(model):
    """
    Create a convex relaxation of the model.

    Parameters
    ----------
    model: pyomo.core.base.block._BlockData or pyomo.core.base.PyomoModel.ConcreteModel
        The model or block to be relaxed

    Returns
    -------
    m: pyomo.core.base.block._BlockData or pyomo.core.base.PyomoModel.ConcreteModel
        The relaxed model
    """
    m = clone_shallow_active_flat(model)[0]
    _relax_cloned_model(m)
    return m
