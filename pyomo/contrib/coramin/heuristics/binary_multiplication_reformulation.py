import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData, ScalarVar
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
from pyomo.core.expr import numeric_expr
from typing import Sequence
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numvalue import NumericValue, native_numeric_types
from typing import Union, Sequence
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.coramin.relaxations.mccormick import PWMcCormickRelaxation
from pyomo.contrib.coramin.utils.coramin_enums import RelaxationSide
from pyomo.contrib.coramin.relaxations import iterators
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.gdp.disjunct import AutoLinkedBinaryVar


class BinaryMultiplicationInfo(object):
    def __init__(self, m: _BlockData) -> None:
        self.m = m
        self.root_node = None
        self.constraint_bounds = None


def handle_var(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    if node.is_fixed():
        res = node.value
    else:
        res = node
    return res


def handle_float(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return node


def handle_param(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return node.value


def handle_sum(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return sum(args)


def handle_monomial(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return args[0] * args[1]


def handle_product(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    x, y = args
    xtype = type(x)
    ytype = type(y)

    if xtype in native_numeric_types or ytype in native_numeric_types:
        return x * y
    if (x.is_variable_type() and x.is_binary()) or (
        y.is_variable_type() and y.is_binary()
    ):

        def get_new_rel(m):
            ndx = len(m.relaxations)
            new_rel = PWMcCormickRelaxation()
            setattr(m, f'rel{ndx}', new_rel)
            m.relaxations.append(new_rel)
            return new_rel

        if x.is_variable_type():
            _x = x
        else:
            _x = info.m.vars.add()
            info.m.cons.add(_x == x)
        if y.is_variable_type():
            _y = y
        else:
            _y = info.vars.add()
            info.m.cons.add(_y == y)
        if info.root_node is node:
            clb, cub = info.constraint_bounds
            if clb == cub and clb is not None:
                rel = get_new_rel(info.m)
                rel.build(
                    _x, _y, clb, relaxation_side=RelaxationSide.BOTH, safety_tol=0
                )
            else:
                if clb is not None:
                    rel = get_new_rel(info.m)
                    rel.build(
                        _x, _y, clb, relaxation_side=RelaxationSide.OVER, safety_tol=0
                    )
                if cub is not None:
                    rel = get_new_rel(info.m)
                    rel.build(
                        _x, _y, cub, relaxation_side=RelaxationSide.UNDER, safety_tol=0
                    )
            return None
        else:
            z = info.m.vars.add()
            zlb, zub = compute_bounds_on_expr(node)
            z.setlb(zlb)
            z.setub(zub)
            rel = get_new_rel(info.m)
            rel.build(_x, _y, z, relaxation_side=RelaxationSide.BOTH, safety_tol=0)
            return z
    return x * y


def handle_exp(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.exp(args[0])


def handle_log(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.log(args[0])


def handle_log10(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.log10(args[0])


def handle_sin(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.sin(args[0])


def handle_cos(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.cos(args[0])


def handle_tan(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.tan(args[0])


def handle_asin(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.asin(args[0])


def handle_acos(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.acos(args[0])


def handle_atan(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.atan(args[0])


def handle_sqrt(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.sqrt(args[0])


def handle_abs(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return pe.abs(args[0])


def handle_div(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    x, y = args
    return x / y


def handle_pow(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    x, y = args
    return x**y


def handle_negation(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return -args[0]


def handle_named_expression(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return args[0]


unary_handlers = dict()
unary_handlers['exp'] = handle_exp
unary_handlers['log'] = handle_log
unary_handlers['log10'] = handle_log10
unary_handlers['sin'] = handle_sin
unary_handlers['cos'] = handle_cos
unary_handlers['tan'] = handle_tan
unary_handlers['asin'] = handle_asin
unary_handlers['acos'] = handle_acos
unary_handlers['atan'] = handle_atan
unary_handlers['sqrt'] = handle_sqrt
unary_handlers['abs'] = handle_abs


def handle_unary(
    node: Union[NumericValue, float],
    args: Sequence[Union[NumericValue, float]],
    info: BinaryMultiplicationInfo,
):
    return unary_handlers[node.getname()](node, args, info)


handlers = dict()
handlers[_GeneralVarData] = handle_var
handlers[ScalarVar] = handle_var
handlers[AutoLinkedBinaryVar] = handle_var
handlers[_ParamData] = handle_param
handlers[ScalarParam] = handle_param
handlers[float] = handle_float
handlers[int] = handle_float
handlers[numeric_expr.SumExpression] = handle_sum
handlers[numeric_expr.LinearExpression] = handle_sum
handlers[numeric_expr.MonomialTermExpression] = handle_monomial
handlers[numeric_expr.ProductExpression] = handle_product
handlers[numeric_expr.DivisionExpression] = handle_div
handlers[numeric_expr.PowExpression] = handle_pow
handlers[numeric_expr.NegationExpression] = handle_negation
handlers[numeric_expr.UnaryFunctionExpression] = handle_unary
handlers[numeric_expr.AbsExpression] = handle_abs
handlers[_GeneralExpressionData] = handle_named_expression
handlers[ScalarExpression] = handle_named_expression
handlers[numeric_expr.NPV_SumExpression] = handle_sum
handlers[numeric_expr.NPV_ProductExpression] = handle_product
handlers[numeric_expr.NPV_DivisionExpression] = handle_div
handlers[numeric_expr.NPV_PowExpression] = handle_pow
handlers[numeric_expr.NPV_NegationExpression] = handle_negation
handlers[numeric_expr.NPV_UnaryFunctionExpression] = handle_unary
handlers[numeric_expr.NPV_AbsExpression] = handle_abs


class BinaryMultiplicationWalker(StreamBasedExpressionVisitor):
    def __init__(self, m: _BlockData):
        super().__init__()
        self.info = BinaryMultiplicationInfo(m)

    def exitNode(self, node, data):
        return handlers[node.__class__](node, data, self.info)


def reformulate_binary_multiplication(m: _BlockData):
    """
    The goal of this function is to replace f(x) * y = 0 with
    a McCormick relaxation when y is binary (in which case the
    McCormick relaxation is equivalent).
    """
    r = pe.ConcreteModel()
    r.vars = pe.VarList()
    r.cons = pe.ConstraintList()
    r.relaxations = list()

    walker = BinaryMultiplicationWalker(r)
    info = walker.info

    for c in iterators.nonrelaxation_component_data_objects(
        m, pe.Constraint, active=True, descend_into=True
    ):
        repn = generate_standard_repn(c.body, compute_values=True, quadratic=False)
        if repn.nonlinear_expr is None:
            r.cons.add((c.lb, c.body, c.ub))
        elif not any(v.is_binary() for v in repn.nonlinear_vars):
            r.cons.add((c.lb, c.body, c.ub))
        else:
            info.root_node = c.body
            info.constraint_bounds = (c.lb, c.ub)
            new_body = walker.walk_expression(c.body)
            if new_body is not None:
                r.cons.add((c.lb, new_body, c.ub))

    for obj in iterators.nonrelaxation_component_data_objects(
        m, pe.Objective, active=True, descend_into=True
    ):
        repn = generate_standard_repn(obj.expr, compute_values=True, quadratic=False)
        if repn.nonlinear_expr is None:
            r.obj = pe.Objective(expr=obj.expr, sense=obj.sense)
        elif not any(v.is_binary() for v in repn.nonlinear_vars):
            r.obj = pe.Objective(expr=obj.expr, sense=obj.sense)
        else:
            info.root_node = None
            info.constraint_bounds = None
            new_expr = walker.walk_expression(obj.expr)
            r.obj = pe.Objective(expr=new_expr, sense=obj.sense)

    return r
