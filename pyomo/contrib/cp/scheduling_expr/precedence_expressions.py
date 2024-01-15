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

from pyomo.core.expr.logical_expr import BooleanExpression


class PrecedenceExpression(BooleanExpression):
    def nargs(self):
        return 3

    @property
    def delay(self):
        return self._args_[2]

    def _to_string_impl(self, values, relation):
        delay = int(values[2])
        if delay == 0:
            first = values[0]
        elif delay > 0:
            first = "%s + %s" % (values[0], delay)
        else:
            first = "%s - %s" % (values[0], abs(delay))
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
