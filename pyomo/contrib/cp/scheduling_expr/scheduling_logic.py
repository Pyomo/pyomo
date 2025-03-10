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


from pyomo.core.expr.logical_expr import NaryBooleanExpression, _flattened


class SpanExpression(NaryBooleanExpression):
    """
    Expression over IntervalVars representing that the first arg spans all the
    following args in the schedule. The first arg is absent if and only if all
    the others are absent.

    args:
        args (tuple): Child nodes, of type IntervalVar
    """

    def _to_string(self, values, verbose, smap):
        return "%s.spans(%s)" % (values[0], ", ".join(values[1:]))


class AlternativeExpression(NaryBooleanExpression):
    """
    Expression over IntervalVars representing that if the first arg is present,
    then exactly one of the following args must be present. The first arg is
    absent if and only if all the others are absent.
    """

    # [ESJ 4/4/24]: docplex takes an optional 'cardinality' argument with this
    # too--it generalized to "exactly n" of the intervals have to exist,
    # basically. It would be nice to include this eventually, but this is
    # probably fine for now.

    def _to_string(self, values, verbose, smap):
        return "alternative(%s, [%s])" % (values[0], ", ".join(values[1:]))


class SynchronizeExpression(NaryBooleanExpression):
    """
    Expression over IntervalVars synchronizing the first argument with all of the
    following arguments. That is, if the first argument is present, the remaining
    arguments start and end at the same time as it.
    """

    def _to_string(self, values, verbose, smap):
        return "synchronize(%s, [%s])" % (values[0], ", ".join(values[1:]))


def spans(*args):
    """Creates a new SpanExpression"""

    return SpanExpression(list(_flattened(args)))


def alternative(*args):
    """Creates a new AlternativeExpression"""

    return AlternativeExpression(list(_flattened(args)))


def synchronize(*args):
    """Creates a new SynchronizeExpression"""

    return SynchronizeExpression(list(_flattened(args)))
