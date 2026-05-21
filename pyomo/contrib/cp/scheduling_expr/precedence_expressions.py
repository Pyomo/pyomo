# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.contrib.cp.interval_var import IntervalVarData


class PrecedenceExpression(BooleanExpression):
    __slots__ = ('_args',)
    PRECEDENCE = None

    def __init__(self, args: tuple(IntervalVarData, IntervalVarData, int)):
        self._args = args

    def nargs(self):
        return 3

    @property
    def args(self):
        return self._args

    @property
    def delay(self):
        return self._args[2]

    def _to_string_impl(self, values, relation):
        delay = values[2]
        if delay == '0':
            first = values[0]
        elif delay[0] in '-+':
            first = "%s %s %s" % (values[0], delay[0], delay[1:])
        else:
            first = "%s + %s" % (values[0], delay)
        return "%s %s %s" % (first, relation, values[1])


class BeforeExpression(PrecedenceExpression):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar. We expect them to be
                      (time_that_comes_before, time_that_comes_after, delay).
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values, "<=")


class AtExpression(PrecedenceExpression):
    """
    Base class for all precedence expressions.

    args:
        args (tuple): child nodes of type IntervalVar. We expect them to be
                      (first_time, second_time, delay).
        delay: A (possibly negative) integer value representing the number of
               time periods delay in the precedence relationship
    """

    def _to_string(self, values, verbose, smap):
        return self._to_string_impl(values, "==")
