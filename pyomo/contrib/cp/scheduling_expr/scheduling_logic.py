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
    TODO/
    """

    def _to_string(self, values, verbose, smap):
        return "alternative(%s, [%s])" % (values[0], ", ".join(values[1:]))


def spans(*args):
    """Creates a new SpanExpression"""

    return SpanExpression(list(_flattened(args)))


def alternative(*args):
    """Creates a new AlternativeExpression"""

    return AlternativeExpression(list(_flattened(args)))
