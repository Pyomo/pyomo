import itertools

from six import iterkeys

from pyomo.common import DeveloperError
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.expr.logical_expr import (
    LogicalExpressionBase,
    NotExpression, BinaryLogicalExpression, MultiArgsExpression,
    AndExpression, OrExpression, ImplicationExpression, EquivalenceExpression,
    XorExpression,
    ExactlyExpression, AtMostExpression, AtLeastExpression, Not, Equivalent,
    LogicalOr, Implies, LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor,
)

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.kernel.component_set import ComponentSet

sympy_available = True
try:
    import sympy
    _operatorMap = {
        sympy.Or: LogicalOr,
        sympy.And: LogicalAnd,
        sympy.Implies: Implies,
        sympy.Equivalent: Equivalent,
        sympy.Not: Not,
    }

    _pyomo_operator_map = {
        AndExpression: sympy.And,
        OrExpression: sympy.Or,
        ImplicationExpression: sympy.Implies,
        EquivalenceExpression: sympy.Equivalent,
        XorExpression: sympy.Xor,
        NotExpression: sympy.Not,
    }
except ImportError:
    sympy_available = False


class PyomoSympyLogicalBimap(object):
    def __init__(self):
        self.pyomo2sympy = ComponentMap()
        self.sympy2pyomo = {}
        self.i = 0

    def getPyomoSymbol(self, sympy_object, default=None):
        return self.sympy2pyomo.get(sympy_object, default)

    def getSympySymbol(self, pyomo_object):
        if pyomo_object in self.pyomo2sympy:
            return self.pyomo2sympy[pyomo_object]
        # Pyomo currently ONLY supports Real variables (not complex
        # variables).  If that ever changes, then we will need to
        # revisit hard-coding the symbol type here
        sympy_obj = sympy.Symbol("x%d" % self.i, real=True)
        self.i += 1
        self.pyomo2sympy[pyomo_object] = sympy_obj
        self.sympy2pyomo[sympy_obj] = pyomo_object
        return sympy_obj

    def sympyVars(self):
        return iterkeys(self.sympy2pyomo)


_special_atom_classes = {ExactlyExpression, AtMostExpression, AtLeastExpression}


class Pyomo2SympyVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map, bool_varlist, special_atom_map):
        super(Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map
        self.boolean_variable_list = bool_varlist
        self.special_atom_map = special_atom_map

    def exitNode(self, node, values):
        _op = _pyomo_operator_map.get(node.__class__, None)
        if _op is None:
            return node._apply_operation(values)
        else:
            return _op(*tuple(values))

    def beforeChild(self, node, child):
        #
        # Don't replace native or sympy types
        #
        if type(child) in native_types:
            return False, child
        #
        # We will descend into all expressions...
        #
        if child.is_expression_type():
            if child.__class__ in _special_atom_classes:
                indicator_var = self.boolean_variable_list.add()
                self.special_atom_map[indicator_var] = child
                return False, self.object_map.getSympySymbol(indicator_var)
            else:
                return True, None
        #
        # Replace pyomo variables with sympy variables
        #
        if child.is_potentially_variable():
            return False, self.object_map.getSympySymbol(child)
        #
        # Everything else is a constant...
        #
        return False, value(child)


class Sympy2PyomoVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        super(Sympy2PyomoVisitor, self).__init__()
        self.object_map = object_map

    def enterNode(self, node):
        return (node.args, [])

    def exitNode(self, node, values):
        """ Visit nodes that have been expanded """
        _sympyOp = node
        _op = _operatorMap.get( type(_sympyOp), None )
        if _op is None:
            raise DeveloperError(
                "sympy expression type '%s' not found in the operator "
                "map" % type(_sympyOp) )
        return _op(*tuple(values))

    def beforeChild(self, node, child):
        if not child.args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return False, item
        return True, None


def to_cnf(expr, bool_varlist=None, bool_var_to_special_atoms=None):
    """Converts a Pyomo logical statement to CNF form.

    Note: the atoms AtMostExpression, AtLeastExpression, and ExactlyExpression
    require special treatment if they are not the root node, or if their children are not atoms,
    e.g. AtMost(2, Y1, Y1 | Y2, Y2, Y3)

    As a result, the model may need to be augmented with
    additional boolean indicator variables and logical propositions.
    This function will raise ValueError if a BooleanVarList is
    not provided on which to store the augmented variables,
    and augmented variables are needed.

    This function will return a tuple containing:
    - CNF of original statement, including possible substitutions
    - list of additional CNF statements (for enforcing equivalence)
    - mapping of augmented variables to special atoms (see note above) with only literals as logical arguments

    In addition, the function will have side effects:
    - augmented variables are added to the passed bool_varlist
    - mapping from augmented variables to equivalent special atoms updated with new entries

    """
    if type(expr) in _special_atom_classes:
        # If root node is one of the spcial atoms, recursively convert its
        # children nodes to CNF.
        return _convert_children_to_literals(expr, bool_varlist, bool_var_to_special_atoms)

    # While performing conversion to sympy, substitute new boolean variables for
    # non-root special atoms.
    pyomo_sympy_map = PyomoSympyLogicalBimap()
    bool_var_to_special_atoms = ComponentMap()
    visitor = Pyomo2SympyVisitor(pyomo_sympy_map, bool_varlist, bool_var_to_special_atoms)
    sympy_expr = visitor.walk_expression(expr)

    # For all newly added indicator variables, create statements enforcing equivalence
    new_logic_statements = [Equivalent(bool_var, )]

    cnf_form = sympy.to_cnf(sympy_expr)
    return sympy2pyomo_expression(cnf_form, symbol_map), []  # additional statements


def _convert_children_to_literals(special_atom, bool_varlist, bool_var_to_special_atoms):
    """If the child logical statements are not literals, substitute augmented boolean variables."""
    new_args = [special_atom.args(0)]
    new_statements = []
    need_new_expression = False
    for child in special_atom.args()[1:]:
        if type(child) in native_types or not child.is_expression_type():
            # Child is a literal. Simply append to new argument list.
            new_args.append(child)
        else:
            need_new_expression = True
            child_cnf, child_new_statements, _ = to_cnf(child, bool_varlist, bool_var_to_special_atoms)
            new_args.append(child_cnf)
            new_statements.extend(child_new_statements)
    if need_new_expression:
        new_atom_with_literals = special_atom.__class__(new_args)
        return new_atom_with_literals, new_statements
    else:
        return special_atom, []


def sympyify_expression(expr, bool_varlist):
    """Convert a Pyomo expression to a Sympy expression"""
    #
    # Create the visitor and call it.
    #
    object_map = PyomoSympyLogicalBimap()
    special_atoms = ComponentMap()
    visitor = Pyomo2SympyVisitor(object_map, bool_varlist, special_atoms)
    # is_expr, ans = visitor.beforeChild(None, expr)
    # if not is_expr:
    #     return object_map, ans

    return object_map, visitor.walk_expression(expr), special_atoms


def sympy2pyomo_expression(expr, object_map):
    visitor = Sympy2PyomoVisitor(object_map)
    is_expr, ans = visitor.beforeChild(None, expr)
    if not is_expr:
        return ans
    return visitor.walk_expression(expr)
