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

class Mode(object):
    coopr3_trees = (1,)
    pyomo4_trees = (2,)
mode = _default_mode = Mode.pyomo4_trees
mode = _default_mode = Mode.coopr3_trees

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
