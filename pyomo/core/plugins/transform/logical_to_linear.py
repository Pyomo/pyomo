"""Transformation from BooleanVar and LogicalStatement to Binary and Constraints."""
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import TransformationFactory, BooleanVar, VarList, Binary, LogicalStatement, Block, ConstraintList, \
    native_types, BooleanVarList
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.logical_expr import AndExpression, OrExpression, NotExpression, AtLeastExpression, \
    AtMostExpression, ExactlyExpression, special_logical_atom_types
from pyomo.core.expr.numvalue import native_logical_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.core.kernel.component_map import ComponentMap
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

        new_boolvar_list_name = unique_component_name(model, 'logic_to_linear_augmented_vars')
        new_boolvarlist = BooleanVarList()
        setattr(model, new_boolvar_list_name, new_boolvarlist)
        new_var_list_name = unique_component_name(model, 'logic_to_linear_augmented_vars_asbinary')
        new_varlist = VarList(domain=Binary)
        setattr(model, new_var_list_name, new_varlist)
        indicator_map = ComponentMap()
        cnf_statements = []
        # Convert all logical statements to CNF
        for logic_statement in model.component_data_objects(ctype=LogicalStatement, active=True):
            cnf_statements.extend(to_cnf(logic_statement.body, new_boolvarlist, indicator_map))
            logic_statement.deactivate()

        # Associate new Boolean vars to new binary variables
        for bool_vardata in new_boolvarlist.values():
            new_binary_vardata = new_varlist.add()
            bool_vardata.set_binary_var(new_binary_vardata)

        # Add constraints associated with each CNF statement
        for cnf_statement in cnf_statements:
            for linear_constraint in _cnf_to_linear_constraint_list(cnf_statement):
                new_constrlist.add(expr=linear_constraint)

        # Add bigM associated with special atoms
        for indicator_var, special_atom in indicator_map.items():
            for linear_constraint in _cnf_to_linear_constraint_list(special_atom, indicator_var, new_varlist):
                new_constrlist.add(expr=linear_constraint)
        pass

        # TODO handle logical statements defined in Disjuncts (recursion)
        pass


def _cnf_to_linear_constraint_list(cnf_expr, indicator_var=None, binary_varlist=None):
    return CnfToLinearVisitor(indicator_var, binary_varlist).walk_expression(cnf_expr)


class CnfToLinearVisitor(StreamBasedExpressionVisitor):
    """Convert CNF Logical Statement to linear constraints.

    Expected expression node types: AndExpression, OrExpression, NotExpression,
    AtLeastExpression, AtMostExpression, ExactlyExpression, _BooleanVarData

    """
    def __init__(self, indicator_var, binary_varlist):
        super(CnfToLinearVisitor, self).__init__()
        self._indicator = indicator_var
        self._binary_varlist = binary_varlist

    def exitNode(self, node, values):
        if type(node) == AndExpression:
            return list(values)
        elif type(node) == OrExpression:
            return sum(values) >= 1
        elif type(node) == NotExpression:
            return 1 - values[0]
        # Note: the following special atoms should only be encountered as root nodes.
        # If they are encountered otherwise, something went wrong.
        sum_values = sum(values[1:])
        num_args = node.nargs() - 1  # number of logical arguments
        if self._indicator is None:
            if type(node) == AtLeastExpression:
                return sum_values >= values[0]
            elif type(node) == AtMostExpression:
                return sum_values <= values[0]
            elif type(node) == ExactlyExpression:
                return sum_values == values[0]
        else:
            rhs_lb, rhs_ub = compute_bounds_on_expr(values[0])
            if rhs_lb == float('-inf') or rhs_ub == float('inf'):
                raise ValueError(
                    "Cannnot generate linear constraints for %s([N, *logical_args]) with unbounded N. "
                    "Detected %s <= N <= %s." % (type(node).__name__, rhs_lb, rhs_ub)
                )
            indicator_binary = self._indicator.as_binary()
            if type(node) == AtLeastExpression:
                return [
                    sum_values >= values[0] - rhs_ub * (1 - indicator_binary),
                    sum_values <= values[0] - 1 + (-(rhs_lb - 1) + num_args) * indicator_binary
                ]
            elif type(node) == AtMostExpression:
                return [
                    sum_values <= values[0] + (-rhs_lb + num_args) * (1 - indicator_binary),
                    sum_values >= (values[0] + 1) - (rhs_ub + 1) * indicator_binary
                ]
            elif type(node) == ExactlyExpression:
                less_than_binary = self._binary_varlist.add()
                more_than_binary = self._binary_varlist.add()
                return [
                    sum_values <= values[0] + (-rhs_lb + num_args) * (1 - indicator_binary),
                    sum_values >= values[0] - rhs_ub * (1 - indicator_binary),
                    indicator_binary + less_than_binary + more_than_binary >= 1,
                    sum_values <= values[0] - 1 + (-(rhs_lb - 1) + num_args) * (1 - less_than_binary),
                    sum_values >= values[0] + 1 - (rhs_ub + 1) * (1 - more_than_binary),
                ]
            pass

    def beforeChild(self, node, child):
        if type(node) in special_logical_atom_types and child is node.args[0]:
            return False, child
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
