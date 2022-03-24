from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.common.collections import ComponentSet
from pyomo.core.expr import current as _expr


class _NamedExpressionVisitor(ExpressionValueVisitor):
    def __init__(self):
        self.named_expressions = dict()
        self.variables = dict()
        self.fixed_vars = dict()
        self._external_functions = dict()

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if type(node) in nonpyomo_leaf_types:
            return True, None

        if node.is_variable_type():
            self.variables[id(node)] = node
            if node.is_fixed():
                self.fixed_vars[id(node)] = node
            return True, None

        if node.is_named_expression_type():
            self.named_expressions[id(node)] = node
            return False, None

        if type(node) is _expr.ExternalFunctionExpression:
            self._external_functions[id(node)] = node
            return False, None

        if node.is_expression_type():
            return False, None

        return True, None


_visitor = _NamedExpressionVisitor()


def identify_named_expressions(expr):
    _visitor.__init__()
    _visitor.dfs_postorder_stack(expr)
    return (list(_visitor.named_expressions.values()),
            list(_visitor.variables.values()),
            list(_visitor.fixed_vars.values()),
            list(_visitor._external_functions.values()))
