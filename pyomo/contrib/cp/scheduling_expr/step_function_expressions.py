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

from pyomo.contrib.cp.interval_var import (
    IntervalVar,
    IntervalVarData,
    IntervalVarStartTime,
    IntervalVarEndTime,
)
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression


def _sum_two_units(_self, _other):
    return CumulativeFunction([_self, _other])


def _sum_cumul_and_unit(_cumul, _unit):
    if _cumul.nargs() == len(_cumul._args_):
        # we can just append to the cumul list
        _cumul._args_.append(_unit)
        return CumulativeFunction(_cumul._args_, nargs=len(_cumul._args_))
    else:
        return CumulativeFunction(_cumul.args + [_unit])


def _sum_unit_and_cumul(_unit, _cumul):
    # Nothing to be done: we need to make a new one because we can't prepend to
    # the list of args.
    return CumulativeFunction([_unit] + _cumul.args)


def _sum_cumuls(_self, _other):
    if _self.nargs() == len(_self._args_):
        _self._args_.extend(_other.args)
        return CumulativeFunction(_self._args_, nargs=len(_self._args_))
    else:
        # we have to clone the list of _args_
        return CumulativeFunction(_self.args + _other.args)


def _subtract_two_units(_self, _other):
    return CumulativeFunction([_self, NegatedStepFunction((_other,))])


def _subtract_cumul_and_unit(_cumul, _unit):
    if _cumul.nargs() == len(_cumul._args_):
        # we can just append to the cumul list
        _cumul._args_.append(NegatedStepFunction((_unit,)))
        return CumulativeFunction(_cumul._args_, nargs=len(_cumul._args_))
    else:
        return CumulativeFunction(_cumul.args + [NegatedStepFunction((_unit,))])


def _subtract_unit_and_cumul(_unit, _cumul):
    # Nothing to be done: we need to make a new one because we can't prepend to
    # the list of args.
    return CumulativeFunction(
        [_unit] + [NegatedStepFunction((a,)) for a in _cumul.args]
    )


def _subtract_cumuls(_self, _other):
    if _self.nargs() == len(_self._args_):
        _self._args_.extend([NegatedStepFunction((a,)) for a in _other.args])
        return CumulativeFunction(_self._args_, nargs=len(_self._args_))
    else:
        # we have to clone the list of _args_
        return CumulativeFunction(
            _self.args + [NegatedStepFunction((a,)) for a in _other.args]
        )


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

    @property
    def args(self):
        return self._args_[: self.nargs()]


class Pulse(StepFunction):
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

    __slots__ = '_args_'

    def __init__(self, args=None, interval_var=None, height=None):
        if args:
            if any(arg is not None for arg in (interval_var, height)):
                raise ValueError(
                    "Cannot specify both args and any of {interval_var, height}"
                )
            # Make sure this is a list because we may add to it if this is
            # summed with other StepFunctions
            self._args_ = [arg for arg in args]
        else:
            self._args_ = [interval_var, height]

        interval_var = self._args_[0]
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
        return self._args_[0]

    @property
    def _height(self):
        return self._args_[1]

    def nargs(self):
        return 2

    def _to_string(self, values, verbose, smap):
        return "Pulse(%s, height=%s)" % (values[0], values[1])


class Step(StepFunction):
    """
    A step function specified by a time point and an integer height that
    has value 0 before the time point and takes the value specified by the
    'height' after the time point.

    Args:
        time (IntervalVarTimePoint or int): the time point at which the step
            function becomes non-zero
        height (int): The value of the step function after the time point
    """

    __slots__ = '_args_'

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


class StepBase(StepFunction):
    __slots__ = '_args_'

    def __init__(self, args):
        # Make sure this is a list because we may add to it if this is summed
        # with otther StepFunctions
        self._args_ = [arg for arg in args]

    @property
    def _time(self):
        return self._args_[0]

    @property
    def _height(self):
        return self._args_[1]

    def nargs(self):
        return 2

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (values[0], values[1])


class StepAt(StepBase):
    __slots__ = ()


class StepAtStart(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (self._time.start_time, values[1])


class StepAtEnd(StepBase):
    __slots__ = ()

    def _to_string(self, values, verbose, smap):
        return "Step(%s, height=%s)" % (self._time.end_time, values[1])


class CumulativeFunction(StepFunction):
    """
    A sum of elementary step functions (Pulse and Step), defining a step
    function over time. (Often used to model resource constraints.)

    Args:
        args (list or tuple): Child elementary step functions of this node
    """

    __slots__ = ('_args_', '_nargs')

    def __init__(self, args, nargs=None):
        # We make sure args are a list because we might add to them later, if
        # this is summed with another cumulative function.
        self._args_ = [arg for arg in args]
        if nargs is None:
            self._nargs = len(args)
        else:
            self._nargs = nargs

    def nargs(self):
        return self._nargs

    def _to_string(self, values, verbose, smap):
        s = ""
        for i, arg in enumerate(self.args):
            if isinstance(arg, NegatedStepFunction):
                s += str(arg) + ' '
            else:
                s += "+ %s "[2 * (i == 0) :] % str(arg)
        return s[:-1]


class NegatedStepFunction(StepFunction):
    """
    The negated form of an elementary step function: That is, it represents
    subtracting the elementary function's (nonnegative) height rather than
    adding it.

    Args:
       arg (Step or Pulse): Child elementary step function of this node
    """

    __slots__ = '_args_'

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        return 1

    def _to_string(self, values, verbose, smap):
        return "- %s" % values[0]


class AlwaysIn(BooleanExpression):
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

    __slots__ = ()

    def __init__(self, args=None, cumul_func=None, bounds=None, times=None):
        if args:
            if any(arg is not None for arg in {cumul_func, bounds, times}):
                raise ValueError(
                    "Cannot specify both args and any of {cumul_func, bounds, times}"
                )
            self._args_ = args
        else:
            self._args_ = (cumul_func, bounds[0], bounds[1], times[0], times[1])

    def nargs(self):
        return 5

    def _to_string(self, values, verbose, smap):
        return "(%s).within(bounds=(%s, %s), times=(%s, %s))" % (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
        )
