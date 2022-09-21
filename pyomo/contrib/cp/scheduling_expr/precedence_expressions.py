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

class PrecedenceExpression(ExpressionBase, BooleanValue):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def __init__(self, before, after, delay):
        self._args_ = (before, after, delay)

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

    def _to_string_impl(self, before, after, operator):
        delay = self.delay
        if delay == 0:
            first = before
        elif delay > 0:
            first = "%s + %s" % (before, delay)
        else:
            first = "%s - %s" % (before, abs(delay))
        return "%s %s %s" % (first, operator, after)

class StartBeforeStartExpression(PrecedenceExpression):
    """
    Expression representing that one IntervalVar must be scheduled to start
    before another starts

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the start of the first argument
               and the start of the second argument
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.start_time',
                                    values[1] + '.start_time',
                                    "<=")

class StartBeforeEndExpression(PrecedenceExpression):
    """
    Expression representing that one IntervalVar must be scheduled to start
    before another ends

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the start of the first argument
               and the end of the second argument
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.start_time',
                                    values[1] + '.end_time',
                                    "<=")

class EndBeforeStartExpression(PrecedenceExpression):
    """
    Expression representing that one IntervalVar must be scheduled to end
    before another starts

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the end of the first argument
               and the start of the second argument
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.end_time',
                                    values[1] + '.start_time',
                                    "<=")

class EndBeforeEndExpression(PrecedenceExpression):
    """
    Expression representing that one IntervalVar must be scheduled to end
    before another ends

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the end of the first argument
               and the end of the second argument
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.end_time',
                                    values[1] + '.end_time',
                                    "<=")

class StartAtStartExpression(PrecedenceExpression):
    """
    Temporal expression representing that on IntervalVar must be scheduled to
    start in coordination with another's start

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the first argument's start time
               and the second argument's start time
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.start_time',
                                    values[1] + '.start_time',
                                    "==")

class StartAtEndExpression(PrecedenceExpression):
    """
    Temporal expression representing that on IntervalVar must be scheduled to
    start in coordination with another's end

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the first argument's start time
               and the second argument's end time
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.start_time',
                                    values[1] + '.end_time',
                                    "==")

class EndAtStartExpression(PrecedenceExpression):
    """
    Temporal expression representing that on IntervalVar must be scheduled to
    end in coordination with another's start

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the first argument's end time
               and the second argument's start time
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.end_time',
                                    values[1] + '.start_time',
                                    "==")

class EndAtEndExpression(PrecedenceExpression):
    """
    Temporal expression representing that on IntervalVar must be scheduled to
    end in coordination with another's end

    args:
        args (tuple): child nodes of type IntervalVar
        delay: A (possibly negative) integer value representing the number of
               time periods required between the first argument's end time
               and the second argument's end time
    """
    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values[0] + '.end_time',
                                    values[1] + '.end_time',
                                    "==")
