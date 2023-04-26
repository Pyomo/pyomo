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
        pw_linear_function (PiecewiseLinearFunction): piece-wise linear function
            of which this node is an instance.
    """

    __slots__ = ('_pw_linear_function',)
    __autoslot_mappers__ = {'_pw_linear_function': AutoSlots.weakref_mapper}

    def __init__(self, args, pw_linear_function):
        super().__init__(args)
        self._pw_linear_function = weakref_ref(pw_linear_function)

    def nargs(self):
        return len(self._args_)

    @property
    def pw_linear_function(self):
        return self._pw_linear_function()

    def create_node_with_local_data(self, args):
        return self.__class__(args, pw_linear_function=self.pw_linear_function)

    def _to_string(self, values, verbose, smap):
        return "%s(%s)" % (str(self.pw_linear_function), ', '.join(values))

    def polynomial_degree(self):
        return None
