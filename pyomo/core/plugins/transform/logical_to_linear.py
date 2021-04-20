"""Transformation from BooleanVar and LogicalConstraint to Binary and Constraints."""
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core import TransformationFactory, BooleanVar, VarList, Binary, LogicalConstraint, Block, ConstraintList, \
    native_types, BooleanVarList
from pyomo.core.expr.cnf_walker import to_cnf
from pyomo.core.expr.logical_expr import AndExpression, OrExpression, NotExpression, AtLeastExpression, \
    AtMostExpression, ExactlyExpression, special_boolean_atom_types, EqualityExpression, InequalityExpression, \
    RangedExpression
from pyomo.core.expr.numvalue import native_logical_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct


@TransformationFactory.register("core.logical_to_linear", doc="Convert logic to linear constraints")
class LogicalToLinear(IsomorphicTransformation):
    """
    Re-encode logical constraints as linear constraints,
    converting Boolean variables to binary.
    """

    def _apply_to(self, model, **kwds):
        for boolean_var in model.component_objects(ctype=BooleanVar, descend_into=(Block, Disjunct)):
            new_varlist = None
            for bool_vardata in boolean_var.values():
                if new_varlist is None and bool_vardata.get_associated_binary() is None:
                    new_var_list_name = unique_component_name(model, boolean_var.local_name + '_asbinary')
                    new_varlist = VarList(domain=Binary)
                    setattr(model, new_var_list_name, new_varlist)

                if bool_vardata.get_associated_binary() is None:
                    new_binary_vardata = new_varlist.add()
                    bool_vardata.associate_binary_var(new_binary_vardata)
                    if bool_vardata.value is not None:
                        new_binary_vardata.value = int(bool_vardata.value)
                    if bool_vardata.fixed:
                        new_binary_vardata.fix()

        # Process statements in global (entire model) context
        _process_logical_constraints_in_logical_context(model)
        # Process statements that appear in disjuncts
        for disjunct in model.component_data_objects(Disjunct, descend_into=(Block, Disjunct), active=True):
            _process_logical_constraints_in_logical_context(disjunct)


def update_boolean_vars_from_binary(model, integer_tolerance=1e-5):
    """Updates all Boolean variables based on the value of their linked binary variables."""
    for boolean_var in model.component_data_objects(BooleanVar, descend_into=(Block, Disjunct)):
        binary_var = boolean_var.get_associated_binary()
        if binary_var is not None and binary_var.value is not None:
            if abs(binary_var.value - 1) <= integer_tolerance:
                boolean_var.value = True
            elif abs(binary_var.value) <= integer_tolerance:
                boolean_var.value = False
            else:
                raise ValueError("Binary variable has non-{0,1} value: %s = %s" % (binary_var.name, binary_var.value))
            boolean_var.stale = binary_var.stale


def _process_logical_constraints_in_logical_context(context):
    new_xfrm_block_name = unique_component_name(context, 'logic_to_linear')
    new_xfrm_block = Block(doc="Transformation objects for logic_to_linear")
    setattr(context, new_xfrm_block_name, new_xfrm_block)

    new_constrlist = new_xfrm_block.transformed_constraints = ConstraintList()
    new_boolvarlist = new_xfrm_block.augmented_vars = BooleanVarList()
    new_varlist = new_xfrm_block.augmented_vars_asbinary = VarList(domain=Binary)

    indicator_map = ComponentMap()
    cnf_statements = []
    # Convert all logical constraints to CNF
    for logical_constraint in context.component_data_objects(ctype=LogicalConstraint, active=True):
        cnf_statements.extend(to_cnf(logical_constraint.body, new_boolvarlist, indicator_map))
        logical_constraint.deactivate()

    # Associate new Boolean vars to new binary variables
    for bool_vardata in new_boolvarlist.values():
        new_binary_vardata = new_varlist.add()
        bool_vardata.associate_binary_var(new_binary_vardata)

    # Add constraints associated with each CNF statement
    for cnf_statement in cnf_statements:
        for linear_constraint in _cnf_to_linear_constraint_list(cnf_statement):
            new_constrlist.add(expr=linear_constraint)

    # Add bigM associated with special atoms
    # Note: this ad-hoc reformulation may be revisited for tightness in the future.
    old_varlist_length = len(new_varlist)
    for indicator_var, special_atom in indicator_map.items():
        for linear_constraint in _cnf_to_linear_constraint_list(special_atom, indicator_var, new_varlist):
            new_constrlist.add(expr=linear_constraint)

    # Previous step may have added auxiliary binaries. Associate augmented Booleans to them.
    num_new = len(new_varlist) - old_varlist_length
    list_o_vars = list(new_varlist.values())
    if num_new:
        for binary_vardata in list_o_vars[-num_new:]:
            new_bool_vardata = new_boolvarlist.add()
            new_bool_vardata.associate_binary_var(binary_vardata)

    # If added components were not used, remove them.
    # Note: it is ok to simply delete the index_set for these components, because by
    # default, a new set object is generated for each [Thing]List.
    if len(new_constrlist) == 0:
        new_xfrm_block.del_component(new_constrlist.index_set())
        new_xfrm_block.del_component(new_constrlist)
    if len(new_boolvarlist) == 0:
        new_xfrm_block.del_component(new_boolvarlist.index_set())
        new_xfrm_block.del_component(new_boolvarlist)
    if len(new_varlist) == 0:
        new_xfrm_block.del_component(new_varlist.index_set())
        new_xfrm_block.del_component(new_varlist)

    # If block was entirely unused, remove it
    if all(len(l) == 0 for l in (new_constrlist, new_boolvarlist, new_varlist)):
        context.del_component(new_xfrm_block)


def _cnf_to_linear_constraint_list(cnf_expr, indicator_var=None, binary_varlist=None):
    # Screen for constants
    if type(cnf_expr) in native_types or cnf_expr.is_constant():
        if value(cnf_expr) is True:
            return []
        else:
            raise ValueError(
                "Cannot build linear constraint for logical expression with constant value False: %s"
                % cnf_expr)
    if cnf_expr.is_expression_type():
        return CnfToLinearVisitor(indicator_var, binary_varlist).walk_expression(cnf_expr)
    else:
        return [cnf_expr.get_associated_binary() == 1]  # Assume that cnf_expr is a BooleanVar


_numeric_relational_types = {InequalityExpression, EqualityExpression, RangedExpression}


class CnfToLinearVisitor(StreamBasedExpressionVisitor):
    """Convert CNF logical constraint to linear constraints.

    Expected expression node types: AndExpression, OrExpression, NotExpression,
    AtLeastExpression, AtMostExpression, ExactlyExpression, _BooleanVarData

    """
    def __init__(self, indicator_var, binary_varlist):
        super(CnfToLinearVisitor, self).__init__()
        self._indicator = indicator_var
        self._binary_varlist = binary_varlist

    def exitNode(self, node, values):
        if type(node) == AndExpression:
            return list((v if type(v) in _numeric_relational_types else v == 1) for v in values)
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
            indicator_binary = self._indicator.get_associated_binary()
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

    def beforeChild(self, node, child, child_idx):
        if type(node) in special_boolean_atom_types and child is node.args[0]:
            return False, child
        if type(child) in native_logical_types:
            return False, int(child)
        if type(child) in native_types:
            return False, child

        if child.is_expression_type():
            return True, None

        # Only thing left should be _BooleanVarData
        return False, child.get_associated_binary()

    def finalizeResult(self, result):
        if type(result) is list:
            return result
        elif type(result) in _numeric_relational_types:
            return [result]
        else:
            return [result == 1]
