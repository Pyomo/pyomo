#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.autoslots import AutoSlots
from pyomo.core.expr.numeric_expr import NumericExpression
from weakref import ref as weakref_ref

class PiecewiseLinearExpression(NumericExpression):
    """
    A numeric expression node representing a specific instantiation of a
    PiecewiseLinearFunction.

    Args:
        args (list or tuple): Children of this node
        parent (PiecewiseLinearFunction): parent piece-wise linear function
            of which this node is an instance.
        index (non-negative int): this expression's index in the parent's
            '_expressions' object (which is an indexed Expression)
    """
    __autoslot_mappers__ = {'_parent_pw_linear_function':
                            AutoSlots.weakref_mapper}

    def __init__(self, args, parent):
        super().__init__(args)
        self._parent_pw_linear_function = weakref_ref(parent)

    def nargs(self):
        return len(self._args_)

    @property
    def parent_pw_linear_function(self):
        return self._parent_pw_linear_function()

    def _to_string(self, values, verbose, smap):
        return "%s(%s)" % (str(self.parent_pw_linear_function),
                           ', '.join(values))

    def polynomial_degree(self):
        return None
