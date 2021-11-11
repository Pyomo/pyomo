from typing import Dict
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import current as _expr
from pyomo.common.dependencies import attempt_import
from ..cmodel import cmodel, cmodel_available


class PyomoToCModelWalker(ExpressionValueVisitor):
    def __init__(self, pyomo_var_to_cvar_map: Dict, pyomo_param_to_c_param_map: Dict):
        self._pyomo_var_to_cvar_map = pyomo_var_to_cvar_map
        self._pyomo_param_to_cparam_map = pyomo_param_to_c_param_map
        self._constant_pool = set()

    def finalize(self, ans):
        if isinstance(ans, cmodel.Node):
            return ans
        else:
            if len(self._constant_pool) == 0:
                self._constant_pool = set(cmodel.create_constants(100))
            const = self._constant_pool.pop()
            const.value = value(ans)
            return const

    def visit(self, node, values):
        if node.__class__ in _pyomo_to_cmodel_map:
            return _pyomo_to_cmodel_map[node.__class__](node, values, self)
        else:
            raise NotImplementedError('Unsupported expression type: {0}'.format(type(node)))

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, node

        if node.is_variable_type():
            return True, self._pyomo_var_to_cvar_map[id(node)]

        if node.is_parameter_type():
            return True, self._pyomo_param_to_cparam_map[id(node)]

        if node.is_expression_type():
            return False, None

        return True, node


def _pyomo_to_cmodel_ProductExpression(node, values, walker: PyomoToCModelWalker):
    assert len(values) == 2
    return values[0] * values[1]


def _pyomo_to_cmodel_ExternalFunctionExpression(node: _expr.ExternalFunctionExpression, values, walker: PyomoToCModelWalker):
    return cmodel.external_helper(node._fcn._function, values)


def _pyomo_to_cmodel_SumExpression(node, values, walker: PyomoToCModelWalker):
    return sum(values)


def _pyomo_to_cmodel_LinearExpression(node, values, walker: PyomoToCModelWalker):
    res = walker.dfs_postorder_stack(node.constant)
    for c, v in zip(node.linear_coefs, node.linear_vars):
        res += walker.dfs_postorder_stack(c) * walker.dfs_postorder_stack(v)
    return res


def _pyomo_to_cmodel_PowExpression(node, values, walker: PyomoToCModelWalker):
    assert len(values) == 2
    return values[0] ** values[1]


def _pyomo_to_cmodel_DivisionExpression(node, values, walker: PyomoToCModelWalker):
    assert len(values) == 2
    return values[0] / values[1]


def _pyomo_to_cmodel_NegationExpression(node, values, walker: PyomoToCModelWalker):
    assert len(values) == 1
    return -values[0]


def _pyomo_to_cmodel_exp(values):
    assert len(values) == 1
    return cmodel.appsi_exp(values[0])


def _pyomo_to_cmodel_log(values):
    assert len(values) == 1
    return cmodel.appsi_log(values[0])


def _pyomo_to_cmodel_log10(values):
    assert len(values) == 1
    return cmodel.appsi_log10(values[0])


def _pyomo_to_cmodel_sin(values):
    assert len(values) == 1
    return cmodel.appsi_sin(values[0])


def _pyomo_to_cmodel_cos(values):
    assert len(values) == 1
    return cmodel.appsi_cos(values[0])


def _pyomo_to_cmodel_tan(values):
    assert len(values) == 1
    return cmodel.appsi_tan(values[0])


def _pyomo_to_cmodel_asin(values):
    assert len(values) == 1
    return cmodel.appsi_asin(values[0])


def _pyomo_to_cmodel_acos(values):
    assert len(values) == 1
    return cmodel.appsi_acos(values[0])


def _pyomo_to_cmodel_atan(values):
    assert len(values) == 1
    return cmodel.appsi_atan(values[0])


def _pyomo_to_cmodel_sqrt(values):
    assert len(values) == 1
    return values[0] ** 0.5


def _pyomo_to_cmodel_GeneralExpression(node, values, walker: PyomoToCModelWalker):
    assert len(values) == 1
    return values[0]


_unary_map = dict()
_unary_map['exp'] = _pyomo_to_cmodel_exp
_unary_map['log'] = _pyomo_to_cmodel_log
_unary_map['log10'] = _pyomo_to_cmodel_log10
_unary_map['sin'] = _pyomo_to_cmodel_sin
_unary_map['cos'] = _pyomo_to_cmodel_cos
_unary_map['tan'] = _pyomo_to_cmodel_tan
_unary_map['asin'] = _pyomo_to_cmodel_asin
_unary_map['acos'] = _pyomo_to_cmodel_acos
_unary_map['atan'] = _pyomo_to_cmodel_atan
_unary_map['sqrt'] = _pyomo_to_cmodel_sqrt


def _pyomo_to_cmodel_UnaryFunctionExpression(node, values, walker: PyomoToCModelWalker):
    if node.getname() in _unary_map:
        return _unary_map[node.getname()](values)
    else:
        raise NotImplementedError('Unsupported expression type: {0}'.format(type(node)))


_pyomo_to_cmodel_map = dict()
_pyomo_to_cmodel_map[_expr.ProductExpression] = _pyomo_to_cmodel_ProductExpression
_pyomo_to_cmodel_map[_expr.ExternalFunctionExpression] = _pyomo_to_cmodel_ExternalFunctionExpression
_pyomo_to_cmodel_map[_expr.DivisionExpression] = _pyomo_to_cmodel_DivisionExpression
_pyomo_to_cmodel_map[_expr.PowExpression] = _pyomo_to_cmodel_PowExpression
_pyomo_to_cmodel_map[_expr.SumExpression] = _pyomo_to_cmodel_SumExpression
_pyomo_to_cmodel_map[_expr.MonomialTermExpression] = _pyomo_to_cmodel_ProductExpression
_pyomo_to_cmodel_map[_expr.NegationExpression] = _pyomo_to_cmodel_NegationExpression
_pyomo_to_cmodel_map[_expr.UnaryFunctionExpression] = _pyomo_to_cmodel_UnaryFunctionExpression
_pyomo_to_cmodel_map[_expr.LinearExpression] = _pyomo_to_cmodel_LinearExpression

_pyomo_to_cmodel_map[_expr.NPV_ProductExpression] = _pyomo_to_cmodel_ProductExpression
_pyomo_to_cmodel_map[_expr.NPV_DivisionExpression] = _pyomo_to_cmodel_DivisionExpression
_pyomo_to_cmodel_map[_expr.NPV_PowExpression] = _pyomo_to_cmodel_PowExpression
_pyomo_to_cmodel_map[_expr.NPV_SumExpression] = _pyomo_to_cmodel_SumExpression
_pyomo_to_cmodel_map[_expr.NPV_NegationExpression] = _pyomo_to_cmodel_NegationExpression
_pyomo_to_cmodel_map[_expr.NPV_UnaryFunctionExpression] = _pyomo_to_cmodel_UnaryFunctionExpression

_pyomo_to_cmodel_map[_GeneralExpressionData] = _pyomo_to_cmodel_GeneralExpression
_pyomo_to_cmodel_map[ScalarExpression] = _pyomo_to_cmodel_GeneralExpression


