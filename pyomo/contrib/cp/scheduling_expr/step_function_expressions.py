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

def _generate_sum_expression(_self, _other):
    if isinstance(_other, StepFunction):
        if _self.nargs() == len(_self._args_):
            _self._args_.extend(_other.args)
            return CumulativeFunction(_self._args_, nargs=len(_self._args_))
        else:
            # we have to clone the list of _args_
            return CumulativeFunction(_self._args_ + _other.args)
    else:
        raise TypeError("Cannot add object of class %s to a "
                        "StepFunction" % _other.__class__)

class StepFunction(object):
    """
    The base class for the step function expression system.
    """
    __slots__ = ()
    #_summable_types = (PulseExpression, StepAtExpression)

    def __add__(self, other):
        return _generate_sum_expression(self, other)

    def __radd__(self, other):
        return _generate_sum_expression(other, self)

    def __iadd__(self, other):
        if isinstance(other, StepFunction):
            if self.nargs() == len(self._args_):
                self._args_.extend(other.args)
                self._nargs = len(self._args_)
            else:
                # have to clone, and then tack on the extra stuff on the end.
                self._args_ = self.args + other.args + \
                              self._args_[self.nargs():]
                self._nargs += other.nargs()
        else:
            raise TypeError("Cannot add object of class %s to a "
                            "StepFunction" % other.__class__)
        return self

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __isub__(self, other):
        pass

    def within(self, cumul_func, bounds, times):
        return AlwaysIn(cumul_func, bounds, times)

    def nargs(self):
        raise NotImplementedError(
            f"Derived expression ({self.__class__}) failed to "
            "implement nargs()")

    @property
    def args(self):
        return self._args_[:self.nargs()]

    # TODO: do we need clone?

class Pulse(StepFunction):
    __slots__ = ('_args_', '_interval_var', '_height')
    def __init__(self, interval_var, height):
        self._args_ = [self]

        self._interval_var = interval_var
        self._height = height

    def __iadd__(self, other):
        # We can't really do this in place because we have to change type.
        return CumulativeFunction([self] + other.args)

    def nargs(self):
        return 1

    def __str__(self):
        return "Pulse(%s, height=%s)" % (self._interval_var.name, self._height)


class Step(StepFunction):
    __slots__ = ('_args_', '_time', '_height')
    def __init__(self, time, height):
        self._args_ = [self]

        self._time = time
        self._height = height

    def __iadd__(self, other):
        # We can't really do this in place because we have to change type.
        return CumulativeFunction([self] + other.args)

    def nargs(self):
        return 1

    def __str__(self):
        return "Step(%s, height=%s)" % (str(self._time), self._height)


class CumulativeFunction(StepFunction):
    def __init__(self, args, nargs=None):
        self._args_ = args
        if nargs is None:
            self._nargs = len(args)
        else:
            self._nargs = nargs

    def nargs(self):
        return self._nargs
        
    def __str__(self):
        return " + ".join([str(arg) for arg in self.args])


class NegatedStepFunction(StepFunction):
    def __init__(self, args, nargs=None):
        self._args_ = args
        if nargs is None:
            self._nargs = len(args)
        else:
            self._nargs = nargs

    def nargs(self):
        return self._nargs
        
    def __str__(self):
        return " - ".join([str(arg) for arg in self.args])


class AlwaysIn(BooleanExpression):
    def __init__(self, cumul_func, bounds, times):
        self._args = (cumul_func, bounds, times)

    def nargs(self):
        return 3

    def _to_string(self, values, verbose, smap):
        return "%s.within(bounds=%s, times=%s)" % (str(values[0]),
                                                   str(values[1]),
                                                   str(values[2]))
