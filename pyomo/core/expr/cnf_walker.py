#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.logical_expr import special_boolean_atom_types
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.expr.sympy_tools import (
    Pyomo2SympyVisitor,
    PyomoSympyBimap,
    sympy,
    sympy2pyomo_expression,
)


class CNF_Pyomo2SympyVisitor(Pyomo2SympyVisitor):
    def __init__(self, object_map, bool_varlist):
        super().__init__(object_map)
        self.boolean_variable_list = bool_varlist
        self.special_atom_map = ComponentMap()

    def beforeChild(self, node, child, child_idx):
        descend, result = super().beforeChild(node, child, child_idx)
        if descend:
            if child.__class__ in special_boolean_atom_types:
                indicator_var = self.boolean_variable_list.add()
                self.special_atom_map[indicator_var] = child
                return False, self.object_map.getSympySymbol(indicator_var)
        return descend, result


def to_cnf(expr, bool_varlist=None, bool_var_to_special_atoms=None):
    """Converts a Pyomo logical constraint to CNF form.

    Note: the atoms AtMostExpression, AtLeastExpression, and
    ExactlyExpression require special treatment if they are not the root
    node, or if their children are not atoms, e.g.

    .. code::

        atmost(2, Y1, Y1 | Y2, Y2, Y3)

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
       - mapping from augmented variables to equivalent special atoms
         (see note above) with only literals as logical arguments

    """
    if type(expr) in special_boolean_atom_types:
        # If root node is one of the special atoms, recursively convert its
        # children nodes to CNF.
        return _convert_children_to_literals(
            expr, bool_varlist, bool_var_to_special_atoms
        )

    # If root node is not an expression, just return it.
    if type(expr) in native_types or not expr.is_expression_type():
        return [expr]

    # While performing conversion to sympy, substitute new boolean variables for
    # non-root special atoms.
    pyomo_sympy_map = PyomoSympyBimap()
    bool_var_to_special_atoms = (
        ComponentMap()
        if bool_var_to_special_atoms is None
        else bool_var_to_special_atoms
    )
    visitor = CNF_Pyomo2SympyVisitor(pyomo_sympy_map, bool_varlist)
    sympy_expr = visitor.walk_expression(expr)

    new_statements = []
    # If visitor encountered any special atoms in non-root node, ensure
    # that their children are literals:
    for indicator_var, special_atom in visitor.special_atom_map.items():
        atom_cnf = _convert_children_to_literals(
            special_atom, bool_varlist, bool_var_to_special_atoms
        )
        bool_var_to_special_atoms[indicator_var] = atom_cnf[0]
        new_statements.extend(atom_cnf[1:])
    cnf_form = sympy.to_cnf(sympy_expr)
    return [
        sympy2pyomo_expression(cnf_form, pyomo_sympy_map)
    ] + new_statements  # additional statements


def _convert_children_to_literals(
    special_atom, bool_varlist, bool_var_to_special_atoms
):
    """If the child logical constraints are not literals, substitute
    augmented boolean variables.

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
                child_cnf = _convert_children_to_literals(
                    child, bool_varlist, bool_var_to_special_atoms
                )
                bool_var_to_special_atoms[new_indicator] = child_cnf[0]
            else:
                child_cnf = to_cnf(
                    new_indicator.equivalent_to(child),
                    bool_varlist,
                    bool_var_to_special_atoms,
                )
                new_statements.append(child_cnf[0])
            new_args.append(new_indicator)
            new_statements.extend(child_cnf[1:])
    if need_new_expression:
        new_atom_with_literals = special_atom.__class__(new_args)
        return [new_atom_with_literals] + new_statements
    else:
        return [special_atom]
