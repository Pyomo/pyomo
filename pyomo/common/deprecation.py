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

"""This module provides utilities for deprecating functionality.

.. autosummary::

   deprecated
   deprecation_warning
   relocated_module
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


def default_deprecation_msg(obj, user_msg, version, remove_in):
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
            _obj = f' class ({_qual.rsplit(".", 1)[0]})'
        elif _qual and _obj:
            _obj += f' ({_qual})'

        user_msg = (
            'This%s has been deprecated and may be removed in a '
            'future release.' % (_obj,)
        )
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
    # Note that _deprecation_docstring is guaranteed to be called by
    # @deprecated in all situations where we would be creating a
    # meaningful deprecation message (classes, functions, and methods),
    # so this is a convenient place to check that the version is
    # specified.
    if version is None:
        raise DeveloperError("@deprecated(): missing 'version' argument")
    return (
        f'{_doc_flag} {version}\n'
        f'   {default_deprecation_msg(obj, msg, None, remove_in)}\n'
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
    # Note: test 'msg is not None' to revert back to the user-supplied
    # message.  Checking the fields above is still useful as it lets us know
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
        for c in reversed(cls.__mro__):
            for f in ('__new__', '__init__'):
                if getattr(c, f, None) is not getattr(cls, f, None):
                    field = f
        setattr(
            cls, field, _wrap_func(getattr(cls, field), msg, logger, version, remove_in)
        )
    return cls


def _wrap_func(func, msg, logger, version, remove_in):
    message = default_deprecation_msg(func, msg, version, remove_in)

    @functools.wraps(
        func, assigned=('__module__', '__name__', '__qualname__', '__annotations__')
    )
    def wrapper(*args, **kwargs):
        cf = _find_calling_frame(1)
        deprecation_warning(message, logger, version='', calling_frame=cf)
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

    return any(mod in sys.modules for mod in ('nose', 'nose2', 'pytest', 'sphinx'))


def deprecation_warning(
    msg, logger=None, version=None, remove_in=None, calling_frame=None
):
    """Standardized formatter for deprecation warnings

    This is a standardized routine for formatting deprecation warnings
    so that things look consistent and "nice".

    Args:
        msg (str): the deprecation message to format

        logger (str): the logger to use for emitting the warning
            (default: the calling pyomo package, or "pyomo")

        version (str): [required] the version in which the decorated
            object was deprecated.  General practice is to set version
            to the current development version (from `pyomo --version`)
            during development and update it to the actual release as
            part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

        calling_frame (frame): the original frame context that triggered
            the deprecation warning.

    Example
    -------
    >>> from pyomo.common.deprecation import deprecation_warning
    >>> deprecation_warning('This functionality is deprecated.', version='1.2.3')
    WARNING: DEPRECATED: This functionality is deprecated.  (deprecated in 1.2.3) ...

    """
    if version is None:
        raise DeveloperError("deprecation_warning() missing 'version' argument")

    if logger is None:
        if calling_frame is not None:
            cf = calling_frame
        else:
            # The relevant module is the one that holds the
            # function/method that called deprecation_warning
            cf = _find_calling_frame(1)
        if cf is not None:
            logger = cf.f_globals.get('__name__', None)
            if logger is not None and not logger.startswith('pyomo'):
                logger = None
        if logger is None:
            logger = 'pyomo'
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    msg = textwrap.fill(
        f'DEPRECATED: {default_deprecation_msg(None, msg, version, remove_in)}',
        width=70,
    )
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

    logger.warning(msg)


if in_testing_environment():
    deprecation_warning.emitted_warnings = None
else:
    deprecation_warning.emitted_warnings = set()


def deprecated(msg=None, logger=None, version=None, remove_in=None):
    """Decorator to indicate that a function, method, or class is deprecated.

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
            to the current development version (from `pyomo --version`)
            during development and update it to the actual release as
            part of the release process.

        remove_in (str): the version in which the decorated object will be
            removed from the code.

    Example
    -------
    >>> from pyomo.common.deprecation import deprecated
    >>> @deprecated(version='1.2.3')
    ... def sample_function(x):
    ...     return 2*x
    >>> sample_function(5)
    WARNING: DEPRECATED: This function (sample_function) has been deprecated and
        may be removed in a future release.  (deprecated in 1.2.3) ...
    10

    """

    def wrap(obj):
        if inspect.isclass(obj):
            return _wrap_class(obj, msg, logger, version, remove_in)
        else:
            return _wrap_func(obj, msg, logger, version, remove_in)

    return wrap


def _import_object(name, target, version, remove_in, msg):
    from importlib import import_module

    modname, targetname = target.rsplit('.', 1)
    _object = getattr(import_module(modname), targetname)
    if msg is None:
        if inspect.isclass(_object):
            _type = 'class'
        elif inspect.isfunction(_object):
            _type = 'function'
        else:
            _type = 'attribute'
        msg = (
            f"the '{name}' {_type} has been moved to '{target}'."
            "  Please update your import."
        )
    deprecation_warning(msg, version=version, remove_in=remove_in)
    return _object


def relocated_module(new_name, msg=None, logger=None, version=None, remove_in=None):
    """Provide a deprecation path for moved / renamed modules

    Upon import, the old module (that called `relocated_module()`) will
    be replaced in `sys.modules` by an alias that points directly to the
    new module.  As a result, the old module should have only two lines
    of executable Python code (the import of `relocated_module` and the
    call to it).

    Parameters
    ----------
    new_name: str
        The new (fully-qualified) module name

    msg: str
        A custom deprecation message.

    logger: str
        The logger to use for emitting the warning (default: the calling
        pyomo package, or "pyomo")

    version: str [required]
        The version in which the module was renamed or moved.  General
        practice is to set version to the current development version
        (from `pyomo --version`) during development and update it to the
        actual release as part of the release process.

    remove_in: str
        The version in which the module will be removed from the code.

    Example
    -------
    >>> from pyomo.common.deprecation import relocated_module
    >>> relocated_module('pyomo.common.deprecation', version='1.2.3')
    WARNING: DEPRECATED: The '...' module has been moved to
        'pyomo.common.deprecation'. Please update your import.
        (deprecated in 1.2.3) ...

    """
    from importlib import import_module

    new_module = import_module(new_name)

    # The relevant module (the one being deprecated) is the one that
    # holds the function/method that called deprecated_module().  The
    # relevant calling frame for the deprecation warning is the first
    # frame in the stack that doesn't look like the importer (i.e., the
    # thing that imported the deprecated module).
    cf = _find_calling_frame(1)
    old_name = cf.f_globals.get('__name__', '<stdin>')
    cf = cf.f_back
    if cf is not None:
        importer = cf.f_back.f_globals['__name__'].split('.')[0]
        while cf is not None and cf.f_globals['__name__'].split('.')[0] == importer:
            cf = cf.f_back
    if cf is None:
        cf = _find_calling_frame(1)

    sys.modules[old_name] = new_module
    if msg is None:
        msg = (
            f"The '{old_name}' module has been moved to '{new_name}'. "
            'Please update your import.'
        )
    deprecation_warning(msg, logger, version, remove_in, cf)


def relocated_module_attribute(
    local, target, version, remove_in=None, msg=None, f_globals=None
):
    """Provide a deprecation path for moved / renamed module attributes

    This function declares that a local module attribute has been moved
    to another location.  For Python 3.7+, it leverages a
    module.__getattr__ method to manage the deferred import of the
    object from the new location (on request), as well as emitting the
    deprecation warning.

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

    msg: str
        If not None, then this specifies a custom deprecation message to
        be emitted when the attribute is accessed from its original
        location.

    """
    if version is None:
        raise DeveloperError("relocated_module_attribute(): missing 'version' argument")
    # Historical note: This method only works for Python >= 3.7.  There
    # were backports to previous Python interpreters, but were removed
    # after SHA 4e04819aaeefc2c08b7718460918885e12343451
    if f_globals is None:
        f_globals = inspect.currentframe().f_back.f_globals
        if f_globals['__name__'].startswith('importlib.'):
            raise DeveloperError(
                "relocated_module_attribute() called from a cythonized "
                "module without passing f_globals"
            )
    _relocated = f_globals.get('__relocated_attrs__', None)
    if _relocated is None:
        f_globals['__relocated_attrs__'] = _relocated = {}
        _mod_getattr = f_globals.get('__getattr__', None)

        def __getattr__(name):
            info = _relocated.get(name, None)
            if info is not None:
                target_obj = _import_object(name, *info)
                f_globals[name] = target_obj
                return target_obj
            elif _mod_getattr is not None:
                return _mod_getattr(name)
            raise AttributeError(
                "module '%s' has no attribute '%s'" % (f_globals['__name__'], name)
            )

        f_globals['__getattr__'] = __getattr__
    _relocated[local] = (target, version, remove_in, msg)


class RenamedClass(type):
    """Metaclass to provide a deprecation path for renamed classes

    This metaclass provides a mechanism for renaming old classes while
    still preserving isinstance / issubclass relationships.

    Examples
    --------
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

    Finally, `isinstance` and `issubclass` still work, for example:

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
                cls.__renamed__warning__("Instantiating class '%s'." % (cls.__name__,))
                return new_class(*args, **kwargs)

            classdict['__new__'] = __renamed__new__

            def __renamed__warning__(msg):
                version = classdict.get('__renamed__version__')
                remove_in = classdict.get('__renamed__remove_in__')
                deprecation_warning(
                    "%s  The class '%s' has been renamed to '%s'."
                    % (msg, name, new_class.__name__),
                    version=version,
                    remove_in=remove_in,
                    calling_frame=_find_calling_frame(1),
                )

            classdict['__renamed__warning__'] = __renamed__warning__

            if not classdict.get('__renamed__version__'):
                raise DeveloperError(
                    "Declaring class '%s' using the RenamedClass metaclass, "
                    "but without specifying the __renamed__version__ class "
                    "attribute" % (name,)
                )

        renamed_bases = []
        for base in bases:
            new_class = getattr(base, '__renamed__new_class__', None)
            if new_class is not None:
                base.__renamed__warning__(
                    "Declaring class '%s' derived from '%s'." % (name, base.__name__)
                )
                base = new_class
                # Flag that this class is derived from a renamed class
                classdict.setdefault('__renamed__new_class__', None)
            # Avoid duplicates (in case someone does a diamond between
            # the renamed class and [a class derived from] the new
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
            if not any(
                hasattr(base, '__renamed__new_class__')
                for mro in itertools.chain.from_iterable(
                    base.__mro__ for base in renamed_bases
                )
            ):
                raise TypeError(
                    "Declaring class '%s' using the RenamedClass metaclass, "
                    "but without specifying the __renamed__new_class__ class "
                    "attribute" % (name,)
                )

        return super().__new__(
            cls, name, tuple(renamed_bases), classdict, *args, **kwargs
        )

    def __instancecheck__(cls, instance):
        # Note: the warning is issued by subclasscheck
        return any(
            cls.__subclasscheck__(c) for c in {type(instance), instance.__class__}
        )

    def __subclasscheck__(cls, subclass):
        if hasattr(cls, '__renamed__warning__'):
            cls.__renamed__warning__(
                "Checking type relative to '%s'." % (cls.__name__,)
            )
        if subclass is cls:
            return True
        elif getattr(cls, '__renamed__new_class__') is not None:
            return issubclass(subclass, getattr(cls, '__renamed__new_class__'))
        else:
            return super().__subclasscheck__(subclass)
