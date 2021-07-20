#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import functools
import inspect

from pyomo.common import DeveloperError

#
# The following decorator is general and should probably be promoted to
# component.py so that we can efficiently handle construction errors on
# scalar components.
#
# TODO: quantify the memory overhead here.  We create (and preserve) a
# locals() dict for *each* method that we wrap.  If that becomes
# significant, we might consider using a single global private
# environment (which would require some thought when managing any
# potential name collisions)
#
_disabled_error = (
    "Cannot %s %s '%s' before it has been constructed (initialized): "
    "'%s' is an attribute on an Abstract component and cannot be "
    "accessed until the component has been fully constructed (converted "
    "to a Concrete component) using AbstractModel.create_instance() or "
    "%s.construct()."
)

def _disable_method(fcn, msg=None):
    _name = fcn.__name__
    if msg is None:
        msg = "access '%s' on" % (_name,)

    # functools.wraps doesn't preserve the function signature until
    # Python 3.4, and even then, does not preserve it accurately (e.g.,
    # calling with the incorrect number of arguments does not generate
    # an error).  For backwards compatability with Python 2.x, we will
    # create a temporary (local) function using exec that matches the
    # function signature passed in and raises an exception
    args = str(inspect.signature(fcn))
    assert args == '(self)' or args.startswith('(self,')

    # lambda comes through with a function name "<lambda>".  We will
    # use exec here to create a function (in a private namespace)
    # that will have the correct name.
    _env = {'_msg': msg, '_name': _name}
    _funcdef = """def %s%s:
        raise RuntimeError("%s" %% (_msg, type(self).__name__,
            self.name, _name, self.name))
""" % (_name, args, _disabled_error)
    exec(_funcdef, _env)
    return functools.wraps(fcn)(_env[_name])


def _disable_property(fcn, msg=None):
    _name = fcn.fget.__name__
    if msg is None:
        _gmsg = "access property '%s' on" % (_name,)
    else:
        _gmsg = msg
    def getter(self, *args, **kwds):
        raise RuntimeError(_disabled_error % (
            _gmsg, type(self).__name__, self.name, _name, self.name))

    if msg is None:
        _smsg = "set property '%s' on" % (_name,)
    else:
        _smsg = msg
    def setter(self, *args, **kwds):
        raise RuntimeError(_disabled_error % (
            _smsg, type(self).__name__, self.name, _name, self.name))

    return property(fget=getter, fset=setter, doc=fcn.__doc__)

def disable_methods(methods):
    """Class decorator to disable methods before construct is called.

    This decorator should be called to create "Abstract" scalar classes
    that override key methods to raise exceptions.  When the construct()
    method is called, the class instance changes type back to the
    original scalar component and the full class functionality is
    restored.  This prevents most class methods from having to begin with
    "`if not self.parent_component()._constructed: raise RuntimeError`"
    """
    def class_decorator(cls):
        assert(len(cls.__bases__) == 1)
        base = cls.__bases__[0]

        def construct(self, data=None):
            if hasattr(self, '_name') and self._name == self.__class__.__name__:
                self._name = base.__name__
            self.__class__ = base
            return base.construct(self, data)
        construct.__doc__ = base.construct.__doc__
        cls.construct = construct

        for method in methods:
            if type(method) is tuple:
                method, msg = method
            else:
                msg = None
            if not hasattr(base, method):
                raise DeveloperError(
                    "Cannot disable method %s on %s: not present on base class"
                    % (method, cls))
            base_method = getattr(base, method)
            if type(base_method) is property:
                setattr(cls, method, _disable_property(base_method, msg))
            else:
                setattr(cls, method, _disable_method(base_method, msg))
        return cls

    return class_decorator
