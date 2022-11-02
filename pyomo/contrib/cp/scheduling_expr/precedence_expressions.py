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

from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.boolean_value import BooleanValue

class BeforeExpression(ExpressionBase, BooleanValue):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def __init__(self, args):
        # We expect args = (before, after, delay)
        self._args_ = args

    def nargs(self):
        return 3

    @property
    def delay(self):
        return self._args_[2]

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Tuple containing the child nodes of this node
        """
        return self._args_

    def _to_string(self, values, verbose, smap):
        delay = int(values[2])
        if delay == 0:
            first = values[0]
        elif delay > 0:
            first = "%s + %s" % (values[0], delay)
        else:
            first = "%s - %s" % (values[0], abs(delay))
        return "%s <= %s" % (first, values[1])


class AtExpression(ExpressionBase, BooleanValue):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def __init__(self, args):
        # We expect args = (first_time, second_time, delay)
        self._args_ = args

    def nargs(self):
        return 3

    @property
    def delay(self):
        return self._args_[2]

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Tuple containing the child nodes of this node
        """
        return self._args_

    def _to_string(self, values, verbose, smap):
        delay = int(values[2])#self.delay
        if delay == 0:
            first = values[0]
        elif delay > 0:
            first = "%s + %s" % (values[0], delay)
        else:
            first = "%s - %s" % (values[0], abs(delay))
        return "%s == %s" % (first, values[1])
