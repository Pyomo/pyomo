"""Transformation from BooleanVar and LogicalStatement to Binary and Constraints."""
from pyomo.common.modeling import unique_component_name
from pyomo.core import TransformationFactory, BooleanVar, VarList, Binary, LogicalStatement, Block, ConstraintList, \
    native_types
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.logical_expr import AndExpression, OrExpression, NotExpression, AtLeastExpression, \
    AtMostExpression, ExactlyExpression
from pyomo.core.expr.numvalue import native_logical_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct


@TransformationFactory.register("core.logical_to_linear", doc="Convert logic to linear constraints")
class LogicalToLinear(IsomorphicTransformation):
    """
    Re-encode logical statements as linear constraints,
    converting Boolean variables to binary.
    """

    def _apply_to(self, model, **kwds):
        for boolean_var in model.component_objects(ctype=BooleanVar, descend_into=(Block, Disjunct)):
            new_varlist = None
            for bool_vardata in boolean_var.values():
                if new_varlist is None and bool_vardata.as_binary() is None:
                    new_var_list_name = unique_component_name(model, boolean_var.local_name + '_asbinary')
                    new_varlist = VarList(domain=Binary)
                    setattr(model, new_var_list_name, new_varlist)

                if bool_vardata.as_binary() is None:
                    new_binary_vardata = new_varlist.add()
                    bool_vardata.set_binary_var(new_binary_vardata)
                    if bool_vardata.value is not None:
                        new_binary_vardata.value = int(bool_vardata.value)
                    if bool_vardata.fixed:
                        new_binary_vardata.fix()

        new_constrlist_name = unique_component_name(model, 'logic_to_linear')
        new_constrlist = ConstraintList()
        setattr(model, new_constrlist_name, new_constrlist)
        for logic_statement in model.component_data_objects(ctype=LogicalStatement, active=True):
            cnf_statement = to_cnf(logic_statement.body)
            for linear_constraint in cnf_to_linear_constraint_list(cnf_statement):
                new_constrlist.add(expr=linear_constraint)
            logic_statement.deactivate()

        # TODO handle logical statements defined in Disjuncts
        pass


def cnf_to_linear_constraint_list(cnf_expr):
    return CnfToLinearVisitor().walk_expression(cnf_expr)


class CnfToLinearVisitor(StreamBasedExpressionVisitor):
    """Convert CNF Logical Statement to linear constraints.

    Expected expression node types: AndExpression, OrExpression, NotExpression,
    AtLeastExpression, AtMostExpression, ExactlyExpression, _BooleanVarData

    """
    def exitNode(self, node, values):
        if type(node) == AndExpression:
            return list(values)
        elif type(node) == OrExpression:
            return sum(values) >= 1
        elif type(node) == NotExpression:
            return 1 - values[0]
        elif type(node) == AtLeastExpression:
            return sum(values[1:]) >= values[0]
        elif type(node) == AtMostExpression:
            return sum(values[1:]) <= values[0]
        elif type(node) == ExactlyExpression:
            return sum(values[1:]) == values[0]

    def beforeChild(self, node, child):
        if type(child) in native_logical_types:
            return False, int(child)
        if type(child) in native_types:
            return False, child

        if child.is_expression_type():
            return True, None

        # Only thing left should be _BooleanVarData
        return False, child.as_binary()

    def finalizeResult(self, result):
        return result if type(result) is list else [result]
