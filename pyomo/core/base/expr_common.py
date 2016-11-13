#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from __future__ import division
from copy import deepcopy
from six import StringIO

class Mode(object):
    coopr3_trees = (1,)
    pyomo4_trees = (2,)
mode = _default_mode = Mode.pyomo4_trees
mode = _default_mode = Mode.coopr3_trees

def clone_expression(exp, substitute=None):
    memo = {'__block_scope__': { id(None): False }}
    if substitute:
        memo.update(substitute)
    return deepcopy(exp, memo)

def _clear_expression_pool():
    from expr_coopr3 import _clear_expression_pool as \
        _clear_expression_pool_coopr3
    from expr_pyomo4 import _clear_expression_pool as \
        _clear_expression_pool_pyomo4
    if mode == Mode.pyomo4_trees:
        _clear_expression_pool_pyomo4()
    else:
        assert mode == Mode.coopr3_trees
        _clear_expression_pool_coopr3()

def chainedInequalityErrorMessage(gre, msg=None):
    if msg is None:
        msg = "Relational expression used in an unexpected Boolean context."
    buf = StringIO()
    gre.chainedInequality.to_string(buf)
    # We are about to raise an exception, so it's OK to reset chainedInequality
    info = gre.call_info
    gre.chainedInequality = None
    gre.call_info =  None

    args = ( str(msg).strip(), buf.getvalue().strip(), info[0], info[1],
             ':\n    %s' % info[3] if info[3] is not None else '.' )
    return """%s

The inequality expression:
    %s
contains non-constant terms (variables) that were evaluated in an
unexpected Boolean context at
  File '%s', line %s%s

Evaluating Pyomo variables in a Boolean context, e.g.
    if expression <= 5:
is generally invalid.  If you want to obtain the Boolean value of the
expression based on the current variable values, explicitly evaluate the
expression using the value() function:
    if value(expression) <= 5:
or
    if value(expression <= 5):
""" % args


ensure_independent_trees = 1
bypass_backreference = 1

TO_STRING_VERBOSE=False

_add = 1
_sub = 2
_mul = 3
_div = 4
_pow = 5
_neg = 6
_abs = 7
_inplace = 10
_unary = _neg

_radd =         -_add
_iadd = _inplace+_add
_rsub =         -_sub
_isub = _inplace+_sub
_rmul =         -_mul
_imul = _inplace+_mul
_rdiv =         -_div
_idiv = _inplace+_div
_rpow =         -_pow
_ipow = _inplace+_pow

_old_etype_strings = {
    'add'  :          _add,
    'radd' :         -_add,
    'iadd' : _inplace+_add,
    'sub'  :          _sub,
    'rsub' :         -_sub,
    'isub' : _inplace+_sub,
    'mul'  :          _mul,
    'rmul' :         -_mul,
    'imul' : _inplace+_mul,
    'div'  :          _div,
    'rdiv' :         -_div,
    'idiv' : _inplace+_div,
    'pow'  :          _pow,
    'rpow' :         -_pow,
    'ipow' : _inplace+_pow,
    'neg'  :          _neg,
    'abs'  :          _abs,
    }

_eq = 0
_le = 1
_lt = 2
