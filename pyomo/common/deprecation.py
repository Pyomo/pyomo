#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""This module provides utilities for deprecating functionality.

.. autosummary::

   deprecated
   deprecation_warning
   relocated_module_attribute
   RenamedClass
"""

import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types

from pyomo.common.errors import DeveloperError

_doc_flag = '.. deprecated::'


def _default_msg(obj, user_msg, version, remove_in):
    """Generate the default deprecation message.

    See deprecated() function for argument details.
    """
    if user_msg is None:
        if inspect.isclass(obj):
            _obj = ' class'
        elif inspect.ismethod(obj):
            _obj = ' method'
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            _obj = ' function'
        else:
            # either @deprecated() an unknown type or called from
            # deprecation_warning()
            _obj = ''

        _qual = getattr(obj, '__qualname__', '') or ''
        if _qual.endswith('.__init__') or _qual.endswith('.__new__'):
            _obj = ' class'

        user_msg = 'This%s has been deprecated and may be removed in a ' \
                   'future release.' % (_obj,)
    comment = []
    if version:
        comment.append('deprecated in %s' % (version,))
    if remove_in:
        comment.append('will be removed in (or after) %s' % (remove_in))
    if comment:
        return user_msg + "  (%s)" % (', '.join(comment),)
    else:
        return user_msg


def _deprecation_docstring(obj, msg, version, remove_in):
    if version is None: # or version in ('','tbd','TBD'):
        raise DeveloperError("@deprecated missing initial version")
    return (
        '%s %s\n   %s\n'
        % (_doc_flag, version, _default_msg(obj, msg, None, remove_in))
    )


def _wrap_class(cls, msg, logger, version, remove_in):
    _doc = None
    # Note: __new_member__ is where enum.Enum buries the user's original
    # __new__ method
    for field in ('__new__', '__init__', '__new_member__'):
        _funcDoc = getattr(getattr(cls, field, None), '__doc__', '') or ''
        _flagIdx = _funcDoc.find(_doc_flag)
        if _flagIdx >= 0:
            _doc = _funcDoc[_flagIdx:]
            break
    # Note: test msg is not None to revert back to the user-supplied
    # message.  Checking the fields is still useful as it lets us know
    # if there is already a deprecation message on either new or init.
    if msg is not None or _doc is None:
        _doc = _deprecation_docstring(cls, msg, version, remove_in)
    if cls.__doc__:
        _doc = cls.__doc__ + '\n\n' + _doc
    cls.__doc__ = 'DEPRECATED.\n\n' + _doc

    if _flagIdx < 0:
        # No deprecation message on __init__ or __new__: go through and
        # find the "most derived" implementation of either __new__ or
        # __init__ and wrap that (breaking ties in favor of __init__)
        field = '__init__'
        for c in cls.__mro__:
            for f in ('__init__', '__new__'):
                if getattr(c, f, None) is not getattr(cls, f, None):
                    field = f
        setattr(cls, field, _wrap_func(
            getattr(cls, field), msg, logger, version, remove_in))
    return cls


def _wrap_func(func, msg, logger, version, remove_in):
    message = _default_msg(func, msg, version, remove_in)

    @functools.wraps(func, assigned=(
        '__module__', '__name__', '__qualname__', '__annotations__'))
    def wrapper(*args, **kwargs):
        deprecation_warning(message, logger)
        return func(*args, **kwargs)

    wrapper.__doc__ = 'DEPRECATED.\n\n'
    _doc = func.__doc__ or ''
    if _doc:
        wrapper.__doc__ += _doc + '\n\n'
    wrapper.__doc__ += _deprecation_docstring(func, msg, version, remove_in)
    return wrapper


def _find_calling_frame(module_offset):
    g = [globals()]
    calling_frame = inspect.currentframe().f_back
    while calling_frame is not None:
        if calling_frame.f_globals is g[-1]:
            calling_frame = calling_frame.f_back
        elif len(g) < module_offset:
            g.append(calling_frame.f_globals)
        else:
            break
    return calling_frame


def in_testing_environment():
    """Return True if we are currently running in a "testing" environment

    This currently includes if nose, nose2, pytest, or Sphinx are
    running (imported).

    """

    return any(mod in sys.modules for mod in (
        'nose', 'nose2', 'pytest', 'sphinx'))


def deprecation_warning(msg, logger=None, version=None,
                        remove_in=None, calling_frame=None):
    """Standardized formatter for deprecation warnings

    This is a standardized routine for formatting deprecation warnings
    so that things look consistent and "nice".

    Args:
        msg (str): the deprecation message to format

        logger (str): the logger to use for emitting the warning
            (default: the calling pyomo package, or "pyomo")

        version (str): [required] the version in which the decorated
            object was deprecated.  General practice is to set version
            to '' or 'TBD' during development and update it to the
            actual release as part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

        calling_frame (frame): the original frame context that triggered
            the deprecation warning.

    """
    if logger is None:
        if calling_frame is not None:
            cf = calling_frame
        else:
            # The relevant module is the one that holds the
            # function/method that called deprecation_warning
            cf = _find_calling_frame(1)
        if cf is not None:
            logger = cf.f_globals.get('__package__', None)
            if logger is not None and not logger.startswith('pyomo'):
                logger = None
        if logger is None:
            logger = 'pyomo'

    msg = textwrap.fill(
        'DEPRECATED: %s' % (_default_msg(None, msg, version, remove_in),),
        width=70)
    if calling_frame is None:
        # The useful thing to let the user know is what called the
        # function that generated the deprecation warning.  The current
        # globals() is *this* module.  Walking up the stack to find the
        # frame where the globals() changes tells us the module that is
        # issuing the deprecation warning.  As we assume that *that*
        # module will not trigger its own deprecation warnings, we will
        # walk farther up until the globals() changes again.
        calling_frame = _find_calling_frame(2)
    if calling_frame is not None:
        info = inspect.getframeinfo(calling_frame)
        msg += "\n(called from %s:%s)" % (info.filename.strip(), info.lineno)
        if deprecation_warning.emitted_warnings is not None:
            if msg in deprecation_warning.emitted_warnings:
                return
            deprecation_warning.emitted_warnings.add(msg)

    logging.getLogger(logger).warning(msg)

if in_testing_environment():
    deprecation_warning.emitted_warnings = None
else:
    deprecation_warning.emitted_warnings = set()


def deprecated(msg=None, logger=None, version=None, remove_in=None):
    """Decorator to indicate that a function, method or class is deprecated.

    This decorator will cause a warning to be logged when the wrapped
    function or method is called, or when the deprecated class is
    constructed.  This decorator also updates the target object's
    docstring to indicate that it is deprecated.

    Args:
        msg (str): a custom deprecation message (default: "This
            {function|class} has been deprecated and may be
            removed in a future release.")

        logger (str): the logger to use for emitting the warning
            (default: the calling pyomo package, or "pyomo")

        version (str): [required] the version in which the decorated
            object was deprecated.  General practice is to set version
            to '' or 'TBD' during development and update it to the
            actual release as part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

    """
    def wrap(obj):
        if inspect.isclass(obj):
            return _wrap_class(obj, msg, logger, version, remove_in)
        else:
            return _wrap_func(obj, msg, logger, version, remove_in)
    return wrap


def _import_object(name, target, version, remove_in):
    from importlib import import_module
    modname, targetname = target.rsplit('.',1)
    _object = getattr(import_module(modname), targetname)
    if inspect.isclass(_object):
        _type = 'class'
    elif inspect.isfunction(_object):
        _type = 'function'
    else:
        _type = 'attribute'
    deprecation_warning(
        "the '%s' %s has been moved to '%s'.  Please update your import."
        % (name, _type, target), version=version, remove_in=remove_in)
    return _object

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


class RenamedClass(type):
    """Metaclass to provide a deprecation path for renamed classes

    This metaclass provides a mechanism for renaming old classes while
    still preserving isinstance / issubclass relationships.

    Example
    -------
        >>> from pyomo.common.deprecation import RenamedClass
        >>> class NewClass(object):
        ...     pass
        >>> class OldClass(metaclass=RenamedClass):
        ...     __renamed__new_class__ = NewClass
        ...     __renamed__version__ = '6.0'

        Deriving from the old class generates a warning:

        >>> class DerivedOldClass(OldClass):
        ...     pass
        WARNING: DEPRECATED: Declaring class 'DerivedOldClass' derived from
            'OldClass'. The class 'OldClass' has been renamed to 'NewClass'.
            (deprecated in 6.0) ...

        As does instantiating the old class:

        >>> old = OldClass()
        WARNING: DEPRECATED: Instantiating class 'OldClass'.  The class
            'OldClass' has been renamed to 'NewClass'.  (deprecated in 6.0) ...

        Finally, isinstance and issubclass still work, for example:

        >>> isinstance(old, NewClass)
        True
        >>> class NewSubclass(NewClass):
        ...     pass
        >>> new = NewSubclass()
        >>> isinstance(new, OldClass)
        WARNING: DEPRECATED: Checking type relative to 'OldClass'.  The class
            'OldClass' has been renamed to 'NewClass'.  (deprecated in 6.0) ...
        True

    """
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        new_class = classdict.get('__renamed__new_class__', None)
        if new_class is not None:
            def __renamed__new__(cls, *args, **kwargs):
                cls.__renamed__warning__(
                    "Instantiating class '%s'." % (cls.__name__,))
                return new_class(*args, **kwargs)
            classdict['__new__'] = __renamed__new__

            def __renamed__warning__(msg):
                version = classdict.get('__renamed__version__')
                remove_in = classdict.get('__renamed__remove_in__')
                deprecation_warning(
                    "%s  The class '%s' has been renamed to '%s'." % (
                        msg, name, new_class.__name__),
                    version=version, remove_in=remove_in,
                    calling_frame=_find_calling_frame(1))
            classdict['__renamed__warning__'] = __renamed__warning__

            if '__renamed__version__' not in classdict:
                raise TypeError(
                    "Declaring class '%s' using the RenamedClass metaclass, "
                    "but without specifying the __renamed__version__ class "
                    "attribute" % (name,))

        renamed_bases = []
        for base in bases:
            new_class = getattr(base, '__renamed__new_class__', None)
            if new_class is not None:
                base.__renamed__warning__(
                    "Declaring class '%s' derived from '%s'." % (
                        name, base.__name__,))
                base = new_class
                # Flag that this class is derived from a renamed class
                classdict.setdefault('__renamed__new_class__', None)
            # Avoid duplicates (in case someone does a diamond between
            # the renamed class and [a class dervied from] the new
            # class)
            if base not in renamed_bases:
                renamed_bases.append(base)

        # Add the new class as a "base class" of the renamed class (this
        # makes issubclass(renamed, new_class) work correctly).  As we
        # still never create an actual instance of renamed, this doesn't
        # affect the API)
        if new_class is not None and new_class not in renamed_bases:
            renamed_bases.append(new_class)

        if new_class is None and '__renamed__new_class__' not in classdict:
            if not any(hasattr(base, '__renamed__new_class__') for mro in
                       itertools.chain.from_iterable(
                           base.__mro__ for base in renamed_bases)):
                raise TypeError(
                    "Declaring class '%s' using the RenamedClass metaclass, "
                    "but without specifying the __renamed__new_class__ class "
                    "attribute" % (name,))

        return super().__new__(
            cls, name, tuple(renamed_bases), classdict, *args, **kwargs)

    def __instancecheck__(cls, instance):
        # Note: the warning is issued by subclasscheck
        return any(cls.__subclasscheck__(c)
            for c in {type(instance), instance.__class__})

    def __subclasscheck__(cls, subclass):
        if hasattr(cls, '__renamed__warning__'):
            cls.__renamed__warning__(
                "Checking type relative to '%s'." % (cls.__name__,))
        if subclass is cls:
            return True
        elif getattr(cls, '__renamed__new_class__') is not None:
            return issubclass(subclass, getattr(cls, '__renamed__new_class__'))
        else:
            return super().__subclasscheck__(subclass)
