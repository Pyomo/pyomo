from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import ExpressionValueVisitor, nonpyomo_leaf_types, value, identify_variables
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint


class FBBTException(Exception):
    pass


def _prop_bnds_leaf_to_root_ProductExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.mul(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_SumExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.SumExpression
    bnds_dict: ComponentMap
    """
    arg0 = node.arg(0)
    lb, ub = bnds_dict[arg0]
    for i in range(1, node.nargs()):
        arg = node.arg(i)
        lb2, ub2 = bnds_dict[arg]
        lb, ub = interval.add(lb, ub, lb2, ub2)
    bnds_dict[node] = (lb, ub)


def _prop_bnds_leaf_to_root_PowExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.PowExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.power(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_ReciprocalExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ReciprocalExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.inv(lb1, ub1)


def _prop_bnds_leaf_to_root_NegationExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.sub(0, 0, lb1, ub1)


def _prop_bnds_leaf_to_root_exp(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.exp(lb1, ub1)


def _prop_bnds_leaf_to_root_log(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.log(lb1, ub1)


def _prop_bnds_leaf_to_root_sin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-1, 1)


def _prop_bnds_leaf_to_root_cos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-1, 1)


def _prop_bnds_leaf_to_root_tan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-math.inf, math.inf)


def _prop_bnds_leaf_to_root_asin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-math.inf, math.inf)


def _prop_bnds_leaf_to_root_acos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-math.inf, math.inf)


def _prop_bnds_leaf_to_root_atan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-math.inf, math.inf)


_unary_leaf_to_root_map = dict()
_unary_leaf_to_root_map['exp'] = _prop_bnds_leaf_to_root_exp
_unary_leaf_to_root_map['log'] = _prop_bnds_leaf_to_root_log
_unary_leaf_to_root_map['sin'] = _prop_bnds_leaf_to_root_sin
_unary_leaf_to_root_map['cos'] = _prop_bnds_leaf_to_root_cos
_unary_leaf_to_root_map['tan'] = _prop_bnds_leaf_to_root_tan
_unary_leaf_to_root_map['asin'] = _prop_bnds_leaf_to_root_asin
_unary_leaf_to_root_map['acos'] = _prop_bnds_leaf_to_root_acos
_unary_leaf_to_root_map['atan'] = _prop_bnds_leaf_to_root_atan


def _prop_bnds_leaf_to_root_UnaryFunctionExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    if node.getname() in _unary_leaf_to_root_map:
        _unary_leaf_to_root_map[node.getname()](node, bnds_dict)
    else:
        raise FBBTException('Unsupported expression type for FBBT: {0}'.format(type(node)))


_prop_bnds_leaf_to_root_map = dict()
_prop_bnds_leaf_to_root_map[_expr.ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[_expr.PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[_expr.SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[_expr.MonomialTermExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[_expr.UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression


def _prop_bnds_root_to_leaf_ProductExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1, _ub1 = interval.div(lb0, ub0, lb2, ub2)
    _lb2, _ub2 = interval.div(lb0, ub0, lb1, ub1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    if _lb2 > lb2:
        lb2 = _lb2
    if _ub2 < ub2:
        ub2 = _ub2
    bnds_dict[arg1] = (lb1, ub1)
    bnds_dict[arg2] = (lb2, ub2)


def _prop_bnds_root_to_leaf_SumExpression(node, bnds_dict):
    """
    This implementation is not efficient!!!

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    # first accumulate bounds
    accumulated_bounds = list()
    accumulated_bounds.append(bnds_dict[node.arg(0)])
    lb0, ub0 = bnds_dict[node]
    for i in range(1, node.nargs()):
        accumulated_bounds.append(interval.add(*(accumulated_bounds[i-1]), *(bnds_dict[node.arg(i)])))
    if lb0 > accumulated_bounds[node.nargs() - 1][0]:
        accumulated_bounds[node.nargs() - 1] = (lb0, accumulated_bounds[node.nargs()-1][1])
    if ub0 < accumulated_bounds[node.nargs() - 1][1]:
        accumulated_bounds[node.nargs() - 1] = (accumulated_bounds[node.nargs()-1][0], ub0)

    for i in reversed(range(1, node.nargs())):
        lb0, ub0 = accumulated_bounds[i]
        lb1, ub1 = accumulated_bounds[i-1]
        lb2, ub2 = bnds_dict[node.arg(i)]
        _lb1, _ub1 = interval.sub(lb0, ub0, lb2, ub2)
        _lb2, _ub2 = interval.sub(lb0, ub0, lb1, ub1)
        if _lb1 > lb1:
            lb1 = _lb1
        if _ub1 < ub1:
            ub1 = _ub1
        if _lb2 > lb2:
            lb2 = _lb2
        if _ub2 < ub2:
            ub2 = _ub2
        accumulated_bounds[i-1] = (lb1, ub1)
        bnds_dict[node.arg(i)] = (lb2, ub2)
    lb, ub = bnds_dict[node.arg(0)]
    _lb, _ub = accumulated_bounds[0]
    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[node.arg(0)] = (lb, ub)


def _prop_bnds_root_to_leaf_PowExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1a, _ub1a = interval.exp(*interval.div(*interval.log(lb0, ub0), lb2, ub2))
    _lb1b, _ub1b = interval.sub(0, 0, _lb1a, _ub1a)
    _lb1 = min(_lb1a, _lb1b)
    _ub1 = max(_ub1a, _ub1b)
    _lb2, _ub2 = interval.div(*interval.log(lb0, ub0), *interval.log(lb1, ub1))
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    if _lb2 > lb2:
        lb2 = _lb2
    if _ub2 < ub2:
        ub2 = _ub2
    bnds_dict[arg1] = (lb1, ub1)
    bnds_dict[arg2] = (lb2, ub2)


def _prop_bnds_root_to_leaf_ReciprocalExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.inv(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_NegationExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.sub(0, 0, lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_exp(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.log(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_log(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.exp(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_sin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    pass


def _prop_bnds_root_to_leaf_cos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    pass


def _prop_bnds_root_to_leaf_tan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    pass


def _prop_bnds_root_to_leaf_asin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.arg(0)
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = (-1, 1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_acos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.arg(0)
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = (-1, 1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_atan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    pass


_unary_root_to_leaf_map = dict()
_unary_root_to_leaf_map['exp'] = _prop_bnds_root_to_leaf_exp
_unary_root_to_leaf_map['log'] = _prop_bnds_root_to_leaf_log
_unary_root_to_leaf_map['sin'] = _prop_bnds_root_to_leaf_sin
_unary_root_to_leaf_map['cos'] = _prop_bnds_root_to_leaf_cos
_unary_root_to_leaf_map['tan'] = _prop_bnds_root_to_leaf_tan
_unary_root_to_leaf_map['asin'] = _prop_bnds_root_to_leaf_asin
_unary_root_to_leaf_map['acos'] = _prop_bnds_root_to_leaf_acos
_unary_root_to_leaf_map['atan'] = _prop_bnds_root_to_leaf_atan


def _prop_bnds_root_to_leaf_UnaryFunctionExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    if node.getname() in _unary_root_to_leaf_map:
        _unary_root_to_leaf_map[node.getname()](node, bnds_dict)
    else:
        raise FBBTException('Unsupported expression type for FBBT: {0}'.format(type(node)))


_prop_bnds_root_to_leaf_map = dict()
_prop_bnds_root_to_leaf_map[_expr.ProductExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[_expr.ReciprocalExpression] = _prop_bnds_root_to_leaf_ReciprocalExpression
_prop_bnds_root_to_leaf_map[_expr.PowExpression] = _prop_bnds_root_to_leaf_PowExpression
_prop_bnds_root_to_leaf_map[_expr.SumExpression] = _prop_bnds_root_to_leaf_SumExpression
_prop_bnds_root_to_leaf_map[_expr.MonomialTermExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[_expr.NegationExpression] = _prop_bnds_root_to_leaf_NegationExpression
_prop_bnds_root_to_leaf_map[_expr.UnaryFunctionExpression] = _prop_bnds_root_to_leaf_UnaryFunctionExpression


class _FBBTVisitorLeafToRoot(ExpressionValueVisitor):
    def __init__(self, bnds_dict):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        """
        self.bnds_dict = bnds_dict

    def visit(self, node, values):
        if node.__class__ in _prop_bnds_leaf_to_root_map:
            _prop_bnds_leaf_to_root_map[node.__class__](node, self.bnds_dict)
        else:
            FBBTException('Unsupported expression type for FBBT: {0}'.format(type(node)))
        return None

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.bnds_dict[node] = (node, node)
            return True, None

        if node.is_variable_type():
            lb = value(node.lb)
            ub = value(node.ub)
            if lb is None:
                lb = -math.inf
            if ub is None:
                ub = math.inf
            self.bnds_dict[node] = (lb, ub)
            return True, None

        if not node.is_expression_type():
            assert is_fixed(node)
            val = value(node)
            self.bnds_dict[node] = (val, val)
            return True, None

        return False, None


class _FBBTVisitorRootToLeaf(ExpressionValueVisitor):
    def __init__(self, bnds_dict):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        """
        self.bnds_dict = bnds_dict

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if node.is_variable_type():
            lb, ub = self.bnds_dict[node]
            if lb != -math.inf:
                node.setlb(lb)
            if ub != math.inf:
                node.setub(ub)
            return True, None

        if not node.is_expression_type():
            return True, None

        if node.__class__ in _prop_bnds_root_to_leaf_map:
            _prop_bnds_root_to_leaf_map[node.__class__](node, self.bnds_dict)
        else:
            raise FBBTException('Unsupported expression type for FBBT: {0}'.format(type(node)))

        return False, None


def fbbt_con(con):
    """
    Feasibility based bounds tightening

    Parameters
    ----------
    con: pyomo.core.base.constraint.Constraint
        constraint on which to perform fbbt
    """
    bnds_dict = ComponentMap()

    visitorA = _FBBTVisitorLeafToRoot(bnds_dict)
    visitorA.dfs_postorder_stack(con.body)
    _lb = value(con.lower)
    _ub = value(con.upper)
    if _lb is None:
        _lb = -math.inf
    if _ub is None:
        _ub = math.inf
    lb, ub = bnds_dict[con.body]
    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[con.body] = (lb, ub)
    visitorB = _FBBTVisitorRootToLeaf(bnds_dict)
    visitorB.dfs_postorder_stack(con.body)


def fbbt_block(m, tol=1e-4):
    """

    Parameters
    ----------
    m: pe.block
    tol: float
    """
    var_to_con_map = ComponentMap()
    var_lbs = ComponentMap()
    var_ubs = ComponentMap()
    for c in m.component_data_objects(ctype=Constraint, active=True, descend_into=True, sort=True):
        for v in identify_variables(c.body):
            if v not in var_to_con_map:
                var_to_con_map[v] = list()
            if v.lb is None:
                var_lbs[v] = -math.inf
            else:
                var_lbs[v] = value(v.lb)
            if v.ub is None:
                var_ubs[v] = math.inf
            else:
                var_ubs[v] = value(v.ub)
            var_to_con_map[v].append(c)

    improved_vars = ComponentSet()
    for c in m.component_data_objects(ctype=Constraint, active=True, descend_into=True, sort=True):
        fbbt_con(c)
        for v in identify_variables(c.body):
            if v.lb is not None:
                if value(v.lb) > var_lbs[v] + tol:
                    improved_vars.add(v)
                    var_lbs[v] = value(v.lb)
            if v.ub is not None:
                if value(v.ub) < var_ubs[v] - tol:
                    improved_vars.add(v)
                    var_ubs[v] = value(v.ub)

    while len(improved_vars) > 0:
        v = improved_vars.pop()
        for c in var_to_con_map[v]:
            fbbt_con(c)
            for _v in identify_variables(c.body):
                if _v.lb is not None:
                    if value(_v.lb) > var_lbs[_v] + tol:
                        improved_vars.add(_v)
                        var_lbs[_v] = value(_v.lb)
                if _v.ub is not None:
                    if value(_v.ub) < var_ubs[_v] - tol:
                        improved_vars.add(_v)
                        var_ubs[_v] = value(_v.ub)


def fbbt(comp):
    """

    Parameters
    ----------
    comp: pyomo.core.base.constraint.Constraint or pyomo.core.base.block.Block
    """
    if comp.type() == Constraint:
        fbbt_con(comp)
    elif comp.type() == Block:
        fbbt_block(comp)
    else:
        raise FBBTException('Cannot perform FBBT on objects of type {0}'.format(type(comp)))
