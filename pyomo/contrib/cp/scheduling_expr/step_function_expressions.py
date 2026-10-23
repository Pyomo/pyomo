# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.contrib.cp.interval_var import (
    IntervalVar,
    IntervalVarData,
    IntervalVarStartTime,
    IntervalVarEndTime,
)
from pyomo.core.expr.base import (
    ExpressionBase,
    UnaryExpression_Mixin,
    BinaryExpression_Mixin,
    NaryExpression_Mixin,
    ExtendableNaryExpression_Mixin,
)
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import SumExpression


def _sum_two_units(_self, _other):
    return CumulativeFunction([_self, _other])


def _sum_cumul_and_unit(_cumul, _unit):
    return _cumul._trunc_append(_unit)


def _sum_unit_and_cumul(_unit, _cumul):
    # Nothing to be done: we need to make a new one because we can't prepend to
    # the list of args.
    return CumulativeFunction([_unit] + _cumul.args)


def _sum_cumuls(_self, _other):
    return _self._trunc_extend(_other)


def _subtract_two_units(_self, _other):
    return CumulativeFunction([_self, NegatedStepFunction((_other,))])


def _subtract_cumul_and_unit(_cumul, _unit):
    return _cumul._trunc_append(NegatedStepFunction((_unit,)))


def _negated_units(args):
    return (NegatedStepFunction((unit,)) for unit in args)


def _subtract_unit_and_cumul(_unit, _cumul):
    # Nothing to be done: we need to make a new one because we can't prepend to
    # the list of args.
    return CumulativeFunction(
        [_unit] + [NegatedStepFunction((a,)) for a in _cumul.args]
    )


def _subtract_cumuls(_self, _other):
    return _self._trunc_extend(_other, _negated_units)


def _generate_sum_expression(_self, _other):
    if isinstance(_self, CumulativeFunction):
        if isinstance(_other, CumulativeFunction):
            return _sum_cumuls(_self, _other)
        elif isinstance(_other, StepFunction):
            return _sum_cumul_and_unit(_self, _other)
    elif isinstance(_self, StepFunction):
        if isinstance(_other, CumulativeFunction):
            return _sum_unit_and_cumul(_self, _other)
        elif isinstance(_other, StepFunction):
            return _sum_two_units(_self, _other)
    raise TypeError(
        "Cannot add object of class %s to object of "
        "class %s" % (_other.__class__, _self.__class__)
    )


def _generate_difference_expression(_self, _other):
    if isinstance(_self, CumulativeFunction):
        if isinstance(_other, CumulativeFunction):
            return _subtract_cumuls(_self, _other)
        elif isinstance(_other, StepFunction):
            return _subtract_cumul_and_unit(_self, _other)
    elif isinstance(_self, StepFunction):
        if isinstance(_other, CumulativeFunction):
            return _subtract_unit_and_cumul(_self, _other)
        elif isinstance(_other, StepFunction):
            return _subtract_two_units(_self, _other)
    raise TypeError(
        "Cannot subtract object of class %s from object of "
        "class %s" % (_other.__class__, _self.__class__)
    )


class StepFunction(ExpressionBase):
    """
    The base class for the step function expression system.
    """

    __slots__ = ()
    PRECEDENCE = None

    def __add__(self, other):
        return _generate_sum_expression(self, other)

    def __radd__(self, other):
        # Mathematically this doesn't make a whole lot of sense, but we'll call
        # 0 a function and be happy so that sum() works as expected.
        if other == 0:
            return self
        return _generate_sum_expression(other, self)

    def __iadd__(self, other):
        return _generate_sum_expression(self, other)

    def __sub__(self, other):
        return _generate_difference_expression(self, other)

    def __rsub__(self, other):
        return _generate_difference_expression(other, self)

    def __isub__(self, other):
        return _generate_difference_expression(self, other)

    def within(self, bounds, times):
        return AlwaysIn(cumul_func=self, bounds=bounds, times=times)


class Pulse(BinaryExpression_Mixin, StepFunction):
    """
    A step function specified by an IntervalVar and an integer height that
    has value 0 before the IntervalVar's start_time and after the
    IntervalVar's end time and that takes the value specified by the 'height'
    during the IntervalVar. (These are often used to model resource
    constraints.)

    Args:
        interval_var (IntervalVar): the interval variable over which the
            pulse function is non-zero
        height (int): The value of the pulse function during the time
            interval_var is scheduled
    """

    __slots__ = ('_l_arg', '_r_arg')

    def __init__(self, args=None, interval_var=None, height=None):
        if args:
            if any(arg is not None for arg in (interval_var, height)):
                raise ValueError(
                    "Cannot specify both args and any of {interval_var, height}"
                )
        else:
            args = (interval_var, height)
        super().__init__(args)

        interval_var = self._interval_var
        if (
            not isinstance(interval_var, IntervalVarData)
            or interval_var.ctype is not IntervalVar
        ):
            raise TypeError(
                "The 'interval_var' argument for a 'Pulse' must "
                "be an 'IntervalVar'.\n"
                "Received: %s" % type(interval_var)
            )

    @property
    def _interval_var(self):
        return self._l_arg

    @property
    def _height(self):
        return self._r_arg

    def _to_string(self, values, verbose, smap):
        return "Pulse(%s, height=%s)" % (values[0], values[1])


class Step(StepFunction):
    """A step function specified by a time point and an integer height that
    has value 0 before the time point and takes the value specified by the
    'height' after the time point.

    Note that this class behaves like a factory and does not return
    instances of :class:`Step`.

    Args:
        time (IntervalVarTimePoint or int): the time point at which the step
            function becomes non-zero
        height (int): The value of the step function after the time point

    """

    __slots__ = ()

    def __new__(cls, time, height):
        if isinstance(time, int):
            return StepAt((time, height))
        elif time.ctype is IntervalVarStartTime:
            return StepAtStart((time.get_associated_interval_var(), height))
        elif time.ctype is IntervalVarEndTime:
            return StepAtEnd((time.get_associated_interval_var(), height))
        else:
            raise TypeError(
                "The 'time' argument for a 'Step' must be either "
                "an 'IntervalVarTimePoint' (for example, the "
                "'start_time' or 'end_time' of an IntervalVar) or "
                "an integer time point in the time horizon.\n"
                "Received: %s" % type(time)
            )


class StepBase(BinaryExpression_Mixin, StepFunction):
    __slots__ = ('_l_arg', '_r_arg')

    @property
    def _time(self):
        return self._l_arg

    @property
    def _height(self):
        return self._r_arg


class StepAt(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (values[0], values[1])


class StepAtStart(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (self._time.start_time, values[1])


class StepAtEnd(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (self._time.end_time, values[1])


class CumulativeFunction(ExtendableNaryExpression_Mixin, StepFunction):
    """
    A sum of elementary step functions (Pulse and Step), defining a step
    function over time. (Often used to model resource constraints.)

    Args:
        args (list or tuple): Child elementary step functions of this node
    """

    __slots__ = ('_args', '_nargs')
    PRECEDENCE = SumExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap):
        s = ""
        for i, arg in enumerate(self.args):
            if isinstance(arg, NegatedStepFunction):
                s += str(arg) + ' '
            else:
                s += "+ %s "[2 * (i == 0) :] % str(arg)
        return s[:-1]


class NegatedStepFunction(UnaryExpression_Mixin, StepFunction):
    """
    The negated form of an elementary step function: That is, it represents
    subtracting the elementary function's (nonnegative) height rather than
    adding it.

    Args:
       arg (Step or Pulse): Child elementary step function of this node
    """

    __slots__ = ('_arg',)

    def _to_string(self, values, verbose, smap):
        return "- %s" % values[0]


class AlwaysIn(NaryExpression_Mixin, BooleanExpression):
    """
    An expression representing the constraint that a cumulative function is
    required to take values within a tuple of bounds over a specified time
    interval. (Often used to enforce limits on resource availability.)

    Args:
        cumul_func (CumulativeFunction): Step function being constrained
        bounds (tuple of two integers): Lower and upper bounds to enforce on
            the cumulative function
        times (tuple of two integers): The time interval (start, end) over
            which to enforce the bounds on the values of the cumulative
            function.
    """

    __slots__ = ('_args',)

    def __init__(self, args=None, cumul_func=None, bounds=None, times=None):
        if args:
            if any(arg is not None for arg in (cumul_func, bounds, times)):
                raise ValueError(
                    "Cannot specify both args and any of {cumul_func, bounds, times}"
                )
        else:
            lb, ub = bounds
            start, end = times
            args = (cumul_func, lb, ub, start, end)

        assert len(args) == 5
        super().__init__(args)

    def _to_string(self, values, verbose, smap):
        return "%s.within(bounds=(%s, %s), times=(%s, %s))" % (*values,)
