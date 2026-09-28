# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections import defaultdict
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
    sqrt,
    exp,
    log,
    log10,
    sin,
    cos,
    tan,
)

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')


def _make_square(arg):
    # Convert Square to x ** 2
    return PowExpression((arg[0], 2))


def _make_product(arg):
    # Gurobi's product is n-ary; just use the Pyomo expression system to
    # convert it to the appropriate expression (MonomialTerm, nested
    # product, etc)
    ans = arg[0]
    for term in arg[1:]:
        ans *= term
    return ans


def _make_const(arg):
    return arg


class GurobiPyomoOpMap(defaultdict):
    """Dictionary mapping Gurobi Opcodes to callbacks for creating Pyomo expressions

    We implement this on top of defaultdict so that we can defer the
    resolution of the GurobiPy module until the first time we look up a
    callback in the dictionary.  That lookup will trigger `__missing__`,
    which will populate the dictionary.

    """

    def __missing__(self, key):
        if self:
            raise RuntimeError(
                f"The gurobi-to-pyomo expression converter encountered an "
                f"unexpected (or unsupported) opcode: {op}"
            )

        GRB = gurobipy.GRB
        self.update(
            {
                GRB.OPCODE_PLUS: (SumExpression, ()),
                # GRB.OPCODE_MINUS: , # This is sum of negated term for us
                GRB.OPCODE_UMINUS: (NegationExpression, ()),
                GRB.OPCODE_MULTIPLY: (_make_product, ()),
                GRB.OPCODE_DIVIDE: (DivisionExpression, ()),
                GRB.OPCODE_SQUARE: (_make_square, ()),
                GRB.OPCODE_SQRT: (UnaryFunctionExpression, ('sqrt', sqrt)),
                GRB.OPCODE_EXP: (UnaryFunctionExpression, ('exp', exp)),
                GRB.OPCODE_LOG: (UnaryFunctionExpression, ('log', log)),
                GRB.OPCODE_LOG2: (UnaryFunctionExpression, ('log', log)),
                GRB.OPCODE_LOG10: (UnaryFunctionExpression, ('log10', log10)),
                GRB.OPCODE_POW: (PowExpression, ()),
                GRB.OPCODE_SIN: (UnaryFunctionExpression, ('sin', sin)),
                GRB.OPCODE_COS: (UnaryFunctionExpression, ('cos', cos)),
                GRB.OPCODE_TAN: (UnaryFunctionExpression, ('tan', tan)),
                # GRB.OPCODE_LOGISTIC: We don't have this one.
                GRB.OPCODE_CONSTANT: (_make_const, ()),
            }
        )
        return self[key]


grb_op_to_pyo = GurobiPyomoOpMap()


def grb_nl_to_pyo_expr(opcodes, data, parents, var_map):
    OPCODE_VARIABLE = gurobipy.GRB.OPCODE_VARIABLE
    # Note that we walk the Gurobi data structures in reverse: Gurobi
    # uses a prefix notation, but Pyomo assumes expressions were
    # generated from the leaves to the root.  By reversing the
    # iteration, we are effectively converting the gurobi data structure
    # from prefix to postfix notation.
    for op, dat, parent in zip(reversed(opcodes), reversed(data), reversed(parents)):
        if op == OPCODE_VARIABLE:
            node = var_map[dat]
        else:
            fcn, args = grb_op_to_pyo[op]
            if dat.__class__ is list:
                # Because we are walking the prefix notation in reverse,
                # we are processing arguments right-to-left.  Reverse
                # the arguments to recover the original argument ordering.
                dat.reverse()
            node = fcn(dat, *args)
        # Data starts off holding "-1" for any operators.  If `parent`
        # is anything other than -1, then that is pointing to an
        # operator (whose initial data entry will start as -1).  We can
        # safely replace that entry with the list of processed operands.
        #
        # Note also, the parent of the root node (i.e., parent[0]) is
        # -1, so this will actually overwrite the *last* data entry, but
        # that is also OK (it's an unnecessary operation, but harmless:
        # we know the last entry is *not* an operator, so the data entry
        # will be some (non-list) integer.  Overwriting it is fine now
        # because we are done processing the expresion).
        if data[parent].__class__ is list:
            data[parent].append(node)
        else:
            data[parent] = [node]
    return node
