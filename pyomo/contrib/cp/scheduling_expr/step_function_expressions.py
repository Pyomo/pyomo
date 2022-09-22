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
    IntervalVar, IntervalVarData, IntervalVarStartTime, IntervalVarEndTime)
from pyomo.core.base.component import Component
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.pyomoobject import PyomoObject

def _generate_sum_expression(_self, _other):
    # We check both because we call this function for the reverse operation as
    # well.
    if isinstance(_other, StepFunction) and isinstance(_self, StepFunction):
        if _self.nargs() == len(_self._args_):
            _self._args_.extend(_other.args)
            return CumulativeFunction(_self._args_, nargs=len(_self._args_))
        else:
            # we have to clone the list of _args_
            return CumulativeFunction(_self.args + _other.args)
    else:
        raise TypeError("Cannot add object of class %s to object of class "
                        "%s" % (_other.__class__, _self.__class__))

def _generate_difference_expression(_self, _other):
    # We check both because we call this function for the reverse operation as
    # well.
    if isinstance(_other, StepFunction) and isinstance(_self, StepFunction):
        if _self.nargs() == len(_self._args_):
            _self._args_.extend([NegatedStepFunction(a) for a in _other.args])
            return CumulativeFunction(_self._args_, nargs=len(_self._args_))
        else:
            # we have to clone the list of _args_
            return CumulativeFunction(_self.args + [NegatedStepFunction(a) for a
                                                    in _other.args])
    else:
        raise TypeError("Cannot subtract object of class %s from object of "
                        "class %s" % (_other.__class__, _self.__class__))

class StepFunction(PyomoObject):
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
        return AlwaysIn(self, bounds, times)

    def nargs(self):
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to "
            "implement nargs()")

    @property
    def args(self):
        return self._args_[:self.nargs()]

    # TODO: do we need clone?

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

    __slots__ = ('_args_')
    def __init__(self, interval_var, height):
        self._args_ = [self, interval_var, height]

        if not isinstance(interval_var, IntervalVarData) or \
           interval_var.ctype is not IntervalVar:
            raise TypeError("The 'interval_var' argument for a 'Pulse' must "
                            "be an 'IntervalVar'.\n"
                            "Received: %s" % type(interval_var))

    @property
    def _interval_var(self):
        return self._args_[1]

    @property
    def _height(self):
        return self._args_[2]

    def __iadd__(self, other):
        # We can't really do this in place because we have to change type.
        return CumulativeFunction([self] + other.args)

    def __isub__(self, other):
        # Have to change type
        return CumulativeFunction([self] + [NegatedStepFunction(a) for a in
                                            other.args])

    def nargs(self):
        return 1

    def __str__(self):
        return "Pulse(%s, height=%s)" % (self._interval_var.name, self._height)


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

    __slots__ = ('_args_')

    def __new__(cls, time, height):
        if isinstance(time, int):
            return StepAt(time, height)
        elif time.ctype is IntervalVarStartTime:
            return StepAtStart(time.get_associated_interval_var(), height)
        elif time.ctype is IntervalVarEndTime:
            return StepAtEnd(time.get_associated_interval_var(), height)
        else:
            raise TypeError("The 'time' argument for a 'Step' must be either "
                            "an 'IntervalVarTimePoint' (for example, the "
                            "'start_time' or 'end_time' of an IntervalVar) or "
                            "an integer time point in the time horizon.\n"
                            "Received: %s" % type(time))

class StepBase(StepFunction):
    def __init__(self, time, height):
        self._args_ = [self, time, height]

    @property
    def _time(self):
        return self._args_[1]

    @property
    def _height(self):
        return self._args_[2]

    def __iadd__(self, other):
        # We can't really do this in place because we have to change type.
        return CumulativeFunction([self] + other.args)

    def __isub__(self, other):
        # Have to change type
        return CumulativeFunction([self] + [NegatedStepFunction(a) for a in
                                            other.args])

    def nargs(self):
        return 1

    def __str__(self):
        return "Step(%s, height=%s)" % (str(self._time), self._height)


class StepAt(StepBase):
    pass


class StepAtStart(StepBase):
    def __str__(self):
        return "Step(%s, height=%s)" % (str(self._time.start_time),
                                        self._height)


class StepAtEnd(StepBase):
    def __str__(self):
        return "Step(%s, height=%s)" % (str(self._time.end_time),
                                        self._height)


class CumulativeFunction(StepFunction):
    """
    A sum of elementary step functions (Pulse and Step), defining a step
    function over time. (Often used to model resource constraints.)

    Args:
        args (list or tuple): Child elementary step functions of this node
    """
    def __init__(self, args, nargs=None):
        self._args_ = args
        if nargs is None:
            self._nargs = len(args)
        else:
            self._nargs = nargs

    def nargs(self):
        return self._nargs

    def __str__(self):
        s = ""
        for i, arg in enumerate(self.args):
            if isinstance(arg, NegatedStepFunction):
                s += str(arg) + " "
            else:
                s += "+ %s "[2*(i == 0):] % str(arg)
        return s[:-1]


class NegatedStepFunction(StepFunction):
    """
    The negated form of an elementary step function: That is, it represents
    subtracting the elementary function's (nonnegative) height rather than
    adding it.

    Args:
       arg (Step or Pulse): Child elementary step function of this node
    """
    def __init__(self, arg):
        self._args_ = [arg]

    def nargs(self):
        return 1

    def __str__(self):
        return "- %s" % str(self._args_[0])


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
    def __init__(self, cumul_func, bounds, times):
        self._args_ = (cumul_func, bounds, times)

    def nargs(self):
        return 3

    def __str__(self):
        return "(%s).within(bounds=%s, times=%s)" % (str(self._args_[0]),
                                                     str(self._args_[1]),
                                                     str(self._args_[2]))
