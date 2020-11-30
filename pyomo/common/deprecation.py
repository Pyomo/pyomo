#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Decorator for deprecating functions."""

import logging
import functools
import inspect
import sys
import textwrap
import types

from pyomo.common.errors import DeveloperError


def _default_msg(user_msg, version, remove_in, func=None):
    """Generate the default deprecation message.

    See deprecated() function for argument details.
    """
    if user_msg is None:
        if inspect.isclass(func):
            _obj = ' class'
        elif inspect.isfunction(func):
            _obj = ' function'
        else:
            _obj = ''
        user_msg = 'This%s has been deprecated and may be removed in a ' \
                   'future release.' % (_obj,)
    comment = []
    if version:
        comment.append('deprecated in %s' % (version,))
    if remove_in:
        comment.append('will be removed in %s' % (remove_in))
    if comment:
        user_msg += "  (%s)" % (', '.join(comment))
    return user_msg


def deprecation_warning(msg, logger='pyomo.core', version=None,
                        remove_in=None, calling_frame=None):
    """Standardized formatter for deprecation warnings

    This is a standardized routine for formatting deprecation warnings
    so that things look consistent and "nice".

    Args:
        msg (str): the deprecation message to format
    """
    msg = textwrap.fill(
        'DEPRECATED: %s' % (_default_msg(msg, version, remove_in),),
        width=70)
    if calling_frame is None:
        try:
            g = globals()
            calling_frame = inspect.currentframe().f_back
            while calling_frame is not None and calling_frame.f_globals is g:
                calling_frame = calling_frame.f_back
        except:
            pass
    if calling_frame is not None:
        info = inspect.getframeinfo(calling_frame)
        msg += "\n(called from %s:%s)" % (info.filename.strip(), info.lineno)

    logging.getLogger(logger).warning(msg)


def deprecated(msg=None, logger='pyomo.core', version=None, remove_in=None):
    """Indicate that a function, method or class is deprecated.

    This decorator will cause a warning to be logged when the wrapped
    function or method is called, or when the deprecated class is
    constructed.  This decorator also updates the target object's
    docstring to indicate that it is deprecated.

    Args:
        msg (str): a custom deprecation message (default: "This
            {function|class} has been deprecated and may be
            removed in a future release.")

        logger (str): the logger to use for emitting the warning
            (default: "pyomo.core")

        version (str): [required] the version in which the decorated
            object was deprecated.  General practice is to set version
            to '' or 'TBD' during development and update it to the
            actual release as part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

    """
    if version is None: # or version in ('','tbd','TBD'):
        raise DeveloperError("@deprecated missing initial version")

    def wrap(func):
        message = _default_msg(msg, version, remove_in, func)

        @functools.wraps(func, assigned=('__module__', '__name__'))
        def wrapper(*args, **kwargs):
            deprecation_warning(message, logger)
            return func(*args, **kwargs)

        if func.__doc__ is None:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: %s' % (message,), width=70)
        else:
            wrapper.__doc__ = textwrap.fill(
                'DEPRECATION WARNING: %s' % (message,), width=70) + '\n\n' + \
                textwrap.fill(textwrap.dedent(func.__doc__.strip()))
        return wrapper

    return wrap


def _import_object(name, target, version, remove_in):
    from importlib import import_module
    modname, targetname = target.rsplit('.',1)
    deprecation_warning(
        "the '%s' class has been moved to '%s'" % (name, target),
        version=version, remove_in=remove_in)
    return getattr(import_module(modname), targetname)

class _ModuleGetattrBackport_27(object):
    """Backport for support of module.__getattr__


    Beginning in Python 3.7, modules support the declaration of a
    module-scoped __getattr__ and __dir__ to allow for the dynamic
    resolution of module attributes.  This class wraps the module class
    and implements `__getattr__`.  As it declares no local
    attributes, all module attribute accesses incur a slight runtime
    penalty (one extra function call).

    """
    def __init__(self, module):
        # Wrapped module needs to be a local attribute.  Everything else
        # is delegated to the inner module type
        super(_ModuleGetattrBackport_27, self).__setattr__(
            '_wrapped_module', module)

    def __getattr__(self, name):
        try:
            return getattr(self._wrapped_module, name)
        except AttributeError:
            info = self._wrapped_module.__relocated_attrs__.get(name, None)
            if info is not None:
                target_obj = _import_object(name, *info)
                setattr(self, name, target_obj)
                return target_obj
            raise

    def __dir__(self):
        return dir(self._wrapped_module)

    def __setattr__(self, name, val):
        setattr(self._wrapped_module, name, val)

class _ModuleGetattrBackport_35(types.ModuleType):
    """Backport for support of module.__getattr__

    Beginning in Python 3.7, modules support the declaration of a
    module-scoped __getattr__ and __dir__ to allow for the dynamic
    resolution of module attributes.  This class derives from
    types.ModuleType and implements `__getattr__`.  As it is a direct
    replacement for types.ModuleType (i.e., we can reassign the already
    loaded module to this type, it is more efficient that the
    ModuleGetattrBackport_27 class which must wrap the already loaded
    module.

    """
    def __getattr__(self, name):
        info = self.__relocated_attrs__.get(name, None)
        if info is not None:
            target_obj = _import_object(name, *info)
            setattr(self, name, target_obj)
            return target_obj
        raise AttributeError("module '%s' has no attribute '%s'"
                             % (self.__name__, name))

def relocated_module_attribute(local, target, version, remove_in=None):
    """Provide a deprecation path for moved / renamed module attributes

    This function declares that a local module attribute has been moved
    to another location.  For Python 3.7+, it leverages a
    module.__getattr__ method to manage the deferred import of the
    object from the new location (on request), as well as emitting the
    deprecation warning.

    It contains backports of the __getattr__ functionality for earlier
    versions of Python (although the implementation for 3.5+ is more
    efficient that the implementation for 2.7+)

    Parameters
    ----------
    local: str
        The original (local) name of the relocated attribute
    target: str
        The new absolute import name of the relocated attribute
    version: str
        The Pyomo version when this move was released
        (passed to deprecation_warning)
    remove_in: str
        The Pyomo version when this deprecation path will be removed
        (passed to deprecation_warning)
    """
    _module = sys.modules[inspect.currentframe().f_back.f_globals['__name__']]
    if not hasattr(_module, '__relocated_attrs__'):
        _module.__relocated_attrs__ = {}
        if sys.version_info >= (3,7):
            _relocated = _module.__relocated_attrs__
            _mod_getattr = getattr(_module, '__getattr__', None)
            def __getattr__(name):
                info = _relocated.get(name, None)
                if info is not None:
                    target_obj = _import_object(name, *info)
                    setattr(_module, name, target_obj)
                    return target_obj
                elif _mod_getattr is not None:
                    return _mod_getattr(name)
                raise AttributeError("module '%s' has no attribute '%s'"
                                     % (_module.__name__, name))
            _module.__getattr__ = __getattr__
        elif sys.version_info >= (3,5):
            # If you run across a case where this assertion fails
            # (because someone else has messed with the module type), we
            # could add logic to use the _ModuleGetattrBackport_27 class
            # to wrap the module.  However, as I believe that this will
            # never happen in Pyomo, it is not worth adding unused
            # functionality at this point
            assert _module.__class__ is types.ModuleType
            _module.__class__ = _ModuleGetattrBackport_35
        else: # sys.version_info >= (2,7):
            _module = sys.modules[_module.__name__] \
                      = _ModuleGetattrBackport_27(_module)
    _module.__relocated_attrs__[local] = (target, version, remove_in)
