from six import iterkeys

from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.logical_expr import (
    AndExpression, EquivalenceExpression, equivalent, ImplicationExpression,
    implies, land, lnot, lor, NotExpression,
    OrExpression, special_boolean_atom_types, XorExpression,
)
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

_operatorMap = {}
_pyomo_operator_map = {}

def _configure_sympy(sympy, available):
    if not available:
        return

    _operatorMap.update({
        sympy.Or: lor,
        sympy.And: land,
        sympy.Implies: implies,
        sympy.Equivalent: equivalent,
        sympy.Not: lnot,
    })

    _pyomo_operator_map.update({
        AndExpression: sympy.And,
        OrExpression: sympy.Or,
        ImplicationExpression: sympy.Implies,
        EquivalenceExpression: sympy.Equivalent,
        XorExpression: sympy.Xor,
        NotExpression: sympy.Not,
    })

sympy, _sympy_available = attempt_import('sympy', callback=_configure_sympy)


class _PyomoSympyLogicalBimap(object):
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


class _Pyomo2SympyVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map, bool_varlist):
        sympy.Add  # this ensures _configure_sympy gets run
        super(_Pyomo2SympyVisitor, self).__init__()
        self.object_map = object_map
        self.boolean_variable_list = bool_varlist
        self.special_atom_map = ComponentMap()

    def exitNode(self, node, values):
        _op = _pyomo_operator_map.get(node.__class__, None)
        if _op is None:
            if node.__class__ in special_boolean_atom_types:
                raise ValueError("Encountered special atom class '%s' in root node" % node.__class__)
            return node._apply_operation(values)
        else:
            return _op(*tuple(values))

    def beforeChild(self, node, child, child_idx):
        #
        # Don't replace native or sympy types
        #
        if type(child) in native_types:
            return False, child
        #
        # We will descend into all expressions...
        #
        if child.is_expression_type():
            if child.__class__ in special_boolean_atom_types:
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


class _Sympy2PyomoVisitor(StreamBasedExpressionVisitor):

    def __init__(self, object_map):
        sympy.Add  # this ensures _configure_sympy gets run
        super(_Sympy2PyomoVisitor, self).__init__()
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

    def beforeChild(self, node, child, child_idx):
        if not child.args:
            item = self.object_map.getPyomoSymbol(child, None)
            if item is None:
                item = float(child.evalf())
            return False, item
        return True, None


def to_cnf(expr, bool_varlist=None, bool_var_to_special_atoms=None):
    """Converts a Pyomo logical constraint to CNF form.

    Note: the atoms AtMostExpression, AtLeastExpression, and ExactlyExpression
    require special treatment if they are not the root node, or if their children are not atoms,
    e.g. atmost(2, Y1, Y1 | Y2, Y2, Y3)

    As a result, the model may need to be augmented with
    additional boolean indicator variables and logical propositions.
    This function will raise ValueError if a BooleanVarList is
    not provided on which to store the augmented variables,
    and augmented variables are needed.

    This function will return a list of CNF logical constraints, including:
    - CNF of original statement, including possible substitutions
    - Additional CNF statements (for enforcing equivalence to augmented variables)

    In addition, the function will have side effects:
    - augmented variables are added to the passed bool_varlist
    - mapping from augmented variables to equivalent special atoms (see note above)
      with only literals as logical arguments

    """
    if type(expr) in special_boolean_atom_types:
        # If root node is one of the special atoms, recursively convert its
        # children nodes to CNF.
        return _convert_children_to_literals(expr, bool_varlist, bool_var_to_special_atoms)

    # If root node is not an expression, just return it.
    if type(expr) in native_types or not expr.is_expression_type():
        return [expr]

    # While performing conversion to sympy, substitute new boolean variables for
    # non-root special atoms.
    pyomo_sympy_map = _PyomoSympyLogicalBimap()
    bool_var_to_special_atoms = ComponentMap() if bool_var_to_special_atoms is None else bool_var_to_special_atoms
    visitor = _Pyomo2SympyVisitor(pyomo_sympy_map, bool_varlist)
    sympy_expr = visitor.walk_expression(expr)

    new_statements = []
    # If visitor encountered any special atoms in non-root node, ensure that their children are literals:
    for indicator_var, special_atom in visitor.special_atom_map.items():
        atom_cnf = _convert_children_to_literals(
            special_atom, bool_varlist, bool_var_to_special_atoms)
        bool_var_to_special_atoms[indicator_var] = atom_cnf[0]
        new_statements.extend(atom_cnf[1:])

    cnf_form = sympy.to_cnf(sympy_expr)
    return [_sympy2pyomo_expression(cnf_form, pyomo_sympy_map)] + new_statements  # additional statements


def _convert_children_to_literals(special_atom, bool_varlist, bool_var_to_special_atoms):
    """If the child logical constraints are not literals, substitute augmented boolean variables.

    Same return types as to_cnf() function.

    """
    new_args = [special_atom.args[0]]
    new_statements = []
    need_new_expression = False
    for child in special_atom.args[1:]:
        if type(child) in native_types or not child.is_expression_type():
            # Child is a literal. Simply append to new argument list.
            new_args.append(child)
        else:
            # We need to do a substitution
            need_new_expression = True
            new_indicator = bool_varlist.add()
            if type(child) in special_boolean_atom_types:
                child_cnf = _convert_children_to_literals(child, bool_varlist, bool_var_to_special_atoms)
                bool_var_to_special_atoms[new_indicator] = child_cnf[0]
            else:
                child_cnf = to_cnf(new_indicator.equivalent_to(child), bool_varlist, bool_var_to_special_atoms)
                new_statements.append(child_cnf[0])
            new_args.append(new_indicator)
            new_statements.extend(child_cnf[1:])
    if need_new_expression:
        new_atom_with_literals = special_atom.__class__(new_args)
        return [new_atom_with_literals] + new_statements
    else:
        return [special_atom]


def _sympy2pyomo_expression(expr, object_map):
    visitor = _Sympy2PyomoVisitor(object_map)
    is_expr, ans = visitor.beforeChild(None, expr, None)
    if not is_expr:
        return ans
    return visitor.walk_expression(expr)
