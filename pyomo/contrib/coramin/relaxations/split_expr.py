from pyomo.core.expr import numeric_expr
from pyomo.core.expr.visitor import identify_variables, ExpressionValueVisitor
from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
    NumericValue,
    is_potentially_variable,
)
from typing import MutableMapping, Tuple, Sequence, Union, List


def _flatten_expr_ProductExpression(
    node: numeric_expr.ProductExpression,
    values: Union[Tuple[NumericValue, ...], List[NumericValue]],
):
    arg1, arg2 = values
    arg1_type = type(arg1)
    arg2_type = type(arg2)
    if is_potentially_variable(arg1) and is_potentially_variable(arg2):
        res = numeric_expr.ProductExpression(values)
    elif is_potentially_variable(arg1):
        if arg1_type is numeric_expr.SumExpression:
            res = numeric_expr.SumExpression([arg2 * i for i in arg1.args])
        elif arg1_type is numeric_expr.LinearExpression:
            res = numeric_expr.LinearExpression(
                constant=arg2 * arg1.constant,
                linear_coefs=[arg2 * i for i in arg1.linear_coefs],
                linear_vars=list(arg1.linear_vars),
            )
        else:
            res = numeric_expr.ProductExpression(values)
    elif is_potentially_variable(arg2):
        if arg2_type is numeric_expr.SumExpression:
            res = numeric_expr.SumExpression([arg1 * i for i in arg2.args])
        elif arg2_type is numeric_expr.LinearExpression:
            res = numeric_expr.LinearExpression(
                constant=arg1 * arg2.constant,
                linear_coefs=[arg1 * i for i in arg2.linear_coefs],
                linear_vars=list(arg2.linear_vars),
            )
        else:
            res = numeric_expr.ProductExpression(values)
    else:
        res = numeric_expr.ProductExpression(values)
    return res


def _flatten_expr_SumExpression(
    node: numeric_expr.SumExpression,
    values: Union[Tuple[NumericValue, ...], List[NumericValue]],
):
    all_args = list()
    for arg in values:
        if isinstance(arg, numeric_expr.SumExpression):
            all_args.extend(arg.args)
        elif isinstance(arg, numeric_expr.LinearExpression):
            for c, v in zip(arg.linear_vars, arg.linear_coefs):
                all_args.append(numeric_expr.MonomialTermExpression((c, v)))
            all_args.append(arg.constant)
        else:
            all_args.append(arg)
    return numeric_expr.SumExpression(all_args)


def _flatten_expr_NegationExpression(
    node: numeric_expr.NegationExpression,
    values: Union[Tuple[NumericValue, ...], List[NumericValue]],
):
    assert len(values) == 1
    arg = values[0]
    if isinstance(arg, numeric_expr.SumExpression):
        res = numeric_expr.SumExpression([-i for i in arg.args])
    elif isinstance(arg, numeric_expr.LinearExpression):
        new_args = [
            numeric_expr.MonomialTermExpression((-c, v))
            for c, v in zip(arg.linear_vars, arg.linear_coefs)
        ]
        new_args.append(-arg.constant)
        res = numeric_expr.SumExpression(new_args)
    else:
        res = numeric_expr.NegationExpression((arg,))
    return res


def _flatten_expr_default(
    node: numeric_expr.ExpressionBase,
    values: Union[Tuple[NumericValue, ...], List[NumericValue]],
):
    return node.create_node_with_local_data(tuple(values))


_flatten_expr_map = dict()
_flatten_expr_map[numeric_expr.SumExpression] = _flatten_expr_SumExpression
_flatten_expr_map[numeric_expr.NegationExpression] = _flatten_expr_NegationExpression
_flatten_expr_map[numeric_expr.ProductExpression] = _flatten_expr_ProductExpression


class FlattenExprVisitor(ExpressionValueVisitor):
    def visit(self, node, values):
        node_type = type(node)
        if node_type in _flatten_expr_map:
            return _flatten_expr_map[node_type](node, values)
        else:
            return _flatten_expr_default(node, values)

    def visiting_potential_leaf(self, node):
        node_type = type(node)
        if node_type in nonpyomo_leaf_types:
            return True, node
        elif not node.is_expression_type():
            return True, node
        elif node_type is numeric_expr.LinearExpression:
            return True, node
        else:
            return False, None


def flatten_expr(expr):
    visitor = FlattenExprVisitor()
    return visitor.dfs_postorder_stack(expr)


class Grouper(object):
    def __init__(self):
        self._terms_by_num_var: MutableMapping[
            int, MutableMapping[Tuple[int, ...], NumericValue]
        ] = dict()

    def add_term(self, expr):
        vlist = list(identify_variables(expr=expr, include_fixed=False))
        vlist.sort(key=lambda x: id(x))
        v_ids = tuple(id(v) for v in vlist)
        num_vars = len(vlist)
        if num_vars not in self._terms_by_num_var:
            self._terms_by_num_var[num_vars] = dict()
        if v_ids not in self._terms_by_num_var[num_vars]:
            self._terms_by_num_var[num_vars][v_ids] = expr
        else:
            self._terms_by_num_var[num_vars][v_ids] += expr

    def group(self) -> Sequence[NumericValue]:
        num_var_list = list(self._terms_by_num_var.keys())
        num_var_list.sort(reverse=True)
        for num_vars in num_var_list[1:]:
            for last_num_vars in num_var_list:
                if last_num_vars == num_vars:
                    break
                for v_ids in list(self._terms_by_num_var[num_vars].keys()):
                    v_id_set = set(v_ids)
                    for last_v_ids in list(
                        self._terms_by_num_var[last_num_vars].keys()
                    ):
                        last_v_id_set = set(last_v_ids)
                        if len(v_id_set - last_v_id_set) == 0:
                            self._terms_by_num_var[last_num_vars][
                                last_v_ids
                            ] += self._terms_by_num_var[num_vars][v_ids]
                            del self._terms_by_num_var[num_vars][v_ids]
                            break

        expr_list = list()
        for num_vars in reversed(num_var_list):
            for e in self._terms_by_num_var[num_vars].values():
                expr_list.append(e)

        return expr_list


def split_expr(expr):
    expr = flatten_expr(expr)
    if type(expr) is numeric_expr.SumExpression:
        grouper = Grouper()
        for arg in expr.args:
            grouper.add_term(arg)
        res = grouper.group()
    else:
        res = [expr]
    return res
