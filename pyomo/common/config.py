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
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

"""The Pyomo configuration system.

This module provides classes and methods for building standardized
hierarchical configuration objects based on the :class:`ConfigDict`,
:class:`ConfigList`, and :class:`ConfigValue` classes.

See the :doc:`Pyomo Configuration System
</explanation/developer_utils/config>` for a more detailed
discussion."""

import argparse
import builtins
import enum
import functools
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types

from operator import attrgetter

from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
    deprecated,
    deprecation_warning,
    relocated_module_attribute,
)
from pyomo.common.fileutils import import_file
from pyomo.common.flags import building_documentation, NOTSET
from pyomo.common.formatting import wrap_reStructuredText

logger = logging.getLogger(__name__)

relocated_module_attribute(
    'PYOMO_CONFIG_DIR', 'pyomo.common.envvar.PYOMO_CONFIG_DIR', version='6.1'
)

USER_OPTION = 0
ADVANCED_OPTION = 10
DEVELOPER_OPTION = 20


def Bool(val):
    """Domain validator for bool-like objects.

    This is a more strict domain than ``bool``, as it will error on
    values that do not "look" like a Boolean value (i.e., it accepts
    ``True``, ``False``, 0, 1, and the case insensitive strings
    ``'true'``, ``'false'``, ``'yes'``, ``'no'``, ``'t'``, ``'f'``,
    ``'y'``, and ``'n'``)

    """
    if type(val) is bool:
        return val
    if isinstance(val, str):
        v = val.upper()
        if v in {'TRUE', 'YES', 'T', 'Y', '1'}:
            return True
        if v in {'FALSE', 'NO', 'F', 'N', '0'}:
            return False
    elif int(val) == float(val):
        v = int(val)
        if v in {0, 1}:
            return bool(v)
    raise ValueError("Expected Boolean, but received %s" % (val,))


def Integer(val):
    """Domain validation function admitting integers

    This domain will admit integers, as well as any values that are
    "reasonably exactly" convertible to integers.  This is more strict
    than ``int``, as it will generate errors for floating point values
    that are not integer.

    """
    ans = int(val)
    # We want to give an error for floating point numbers...
    if ans != float(val):
        raise ValueError("Expected integer, but received %s" % (val,))
    return ans


def PositiveInt(val):
    """Domain validation function admitting strictly positive integers

    This domain will admit positive integers (n > 0), as well as any
    types that are convertible to positive integers.

    """
    ans = int(val)
    # We want to give an error for floating point numbers...
    if ans != float(val) or ans <= 0:
        raise ValueError("Expected positive int, but received %s" % (val,))
    return ans


def NegativeInt(val):
    """Domain validation function admitting strictly negative integers

    This domain will admit negative integers (n < 0), as well as any
    types that are convertible to negative integers.

    """
    ans = int(val)
    if ans != float(val) or ans >= 0:
        raise ValueError("Expected negative int, but received %s" % (val,))
    return ans


def NonPositiveInt(val):
    """Domain validation function admitting integers <= 0

    This domain will admit non-positive integers (n <= 0), as well as
    any types that are convertible to non-positive integers.

    """
    ans = int(val)
    if ans != float(val) or ans > 0:
        raise ValueError("Expected non-positive int, but received %s" % (val,))
    return ans


def NonNegativeInt(val):
    """Domain validation function admitting integers >= 0

    This domain will admit non-negative integers (n >= 0), as well as
    any types that are convertible to non-negative integers.

    """
    ans = int(val)
    if ans != float(val) or ans < 0:
        raise ValueError("Expected non-negative int, but received %s" % (val,))
    return ans


def PositiveFloat(val):
    """Domain validation function admitting strictly positive numbers

    This domain will admit positive floating point numbers (n > 0), as
    well as any types that are convertible to positive floating point
    numbers.

    """
    ans = float(val)
    if ans <= 0:
        raise ValueError("Expected positive float, but received %s" % (val,))
    return ans


def NegativeFloat(val):
    """Domain validation function admitting strictly negative numbers

    This domain will admit negative floating point numbers (n < 0), as
    well as any types that are convertible to negative floating point
    numbers.

    """
    ans = float(val)
    if ans >= 0:
        raise ValueError("Expected negative float, but received %s" % (val,))
    return ans


def NonPositiveFloat(val):
    """Domain validation function admitting numbers less than or equal to 0

    This domain will admit non-positive floating point numbers (n <= 0),
    as well as any types that are convertible to non-positive floating
    point numbers.

    """
    ans = float(val)
    if ans > 0:
        raise ValueError("Expected non-positive float, but received %s" % (val,))
    return ans


def NonNegativeFloat(val):
    """Domain validation function admitting numbers greater than or equal to 0

    This domain will admit non-negative floating point numbers (n >= 0),
    as well as any types that are convertible to non-negative floating
    point numbers.

    """
    ans = float(val)
    if ans < 0:
        raise ValueError("Expected non-negative float, but received %s" % (val,))
    return ans


class In(object):
    """In(domain, cast=None)
    Domain validation class admitting a Container of possible values

    This will admit any value that is in the `domain` Container (i.e.,
    Container.__contains__() returns True).  Most common domains are
    list, set, and dict objects.  If specified, incoming values are
    first passed to `cast()` to convert them to the appropriate type
    before looking them up in `domain`.

    Parameters
    ----------
    domain: Container
        The container that specifies the allowable values.  Incoming
        values are passed to ``domain.__contains__()``, and if ``True``
        is returned, the value is accepted and returned.

    cast: Callable, optional
        A callable object.  If specified, incoming values are first
        passed to `cast`, and the resulting object is checked for
        membership in `domain`

    Note
    ----
    For backwards compatibility, `In` accepts `enum.Enum` classes as
    `domain` Containers.  If the domain is an Enum, then the constructor
    returns an instance of `InEnum`.

    """

    def __new__(cls, domain=None, cast=None):
        # Convenience: enum.Enum supported __contains__ through Python
        # 3.7.  If the domain is an Enum and cast is not specified,
        # automatically return an InEnum to handle casting and validation
        if (
            cls is In
            and cast is None
            and inspect.isclass(domain)
            and issubclass(domain, enum.Enum)
        ):
            return InEnum(domain)
        return super(In, cls).__new__(cls)

    def __init__(self, domain, cast=None):
        self._domain = domain
        self._cast = cast

    def __call__(self, value):
        if self._cast is not None:
            v = self._cast(value)
        else:
            v = value
        if v in self._domain:
            return v
        raise ValueError("value %s not in domain %s" % (value, self._domain))

    def domain_name(self):
        _dn = str(self._domain)
        if not _dn or _dn[0] not in '[({':
            return f'In({_dn})'
        else:
            return f'In{_dn}'


class InEnum(object):
    """Domain validation class admitting an enum value/name.

    This will admit any value that is in the specified Enum, including
    Enum members, values, and string names.  The incoming value will be
    automatically cast to an Enum member.

    Parameters
    ----------
    domain: enum.Enum
        The enum that incoming values should be mapped to

    """

    def __init__(self, domain):
        self._domain = domain

    def __call__(self, value):
        try:
            # First check if this is a valid enum value
            return self._domain(value)
        except ValueError:
            # Assume this is a string and look it up
            try:
                return self._domain[value]
            except KeyError:
                pass
        raise ValueError("%r is not a valid %s" % (value, self._domain.__name__))

    def domain_name(self):
        return f'InEnum[{_domain_name(self._domain)}]'


class IsInstance(object):
    """
    Domain validator for type checking.

    Parameters
    ----------
    *bases : tuple of type
        Valid types.
    document_full_base_names : bool, optional
        True to prepend full module qualifier to the name of each
        member of `bases` in ``self.domain_name()`` and/or any
        error messages generated by this object, False otherwise.
    """

    def __init__(self, *bases, document_full_base_names=False):
        assert bases
        self.baseClasses = bases
        self.document_full_base_names = document_full_base_names

    @staticmethod
    def _fullname(klass):
        """
        Get full name of class, including appropriate module qualifier.
        """
        module_name = klass.__module__
        module_qual = "" if module_name == "builtins" else f"{module_name}."
        return f"{module_qual}{klass.__name__}"

    def _get_class_name(self, klass):
        """
        Get name of class. Module qualifier may be included,
        depending on value of `self.document_full_base_names`.
        """
        if self.document_full_base_names:
            return self._fullname(klass)
        else:
            return klass.__name__

    def __call__(self, obj):
        if isinstance(obj, self.baseClasses):
            return obj
        if len(self.baseClasses) > 1:
            class_names = ", ".join(
                f"{self._get_class_name(kls)!r}" for kls in self.baseClasses
            )
            msg = (
                "Expected an instance of one of these types: "
                f"{class_names}, but received value {obj!r} of type "
                f"{self._get_class_name(type(obj))!r}"
            )
        else:
            msg = (
                f"Expected an instance of "
                f"{self._get_class_name(self.baseClasses[0])!r}, "
                f"but received value {obj!r} of type "
                f"{self._get_class_name(type(obj))!r}"
            )
        raise ValueError(msg)

    def domain_name(self):
        class_names = (_domain_name(kls) for kls in self.baseClasses)
        return f"IsInstance[{', '.join(class_names)}]"


class ListOf(object):
    """Domain validator for lists of a specified type

    Parameters
    ----------
    itemtype: type
        The type for each element in the list

    domain: Callable
        A domain validator (callable that takes the incoming value,
        validates it, and returns the appropriate domain type) for each
        element in the list.  If not specified, defaults to the
        `itemtype`.

    string_lexer: Callable
        A preprocessor (lexer) called for all string values.  If
        NOTSET, then strings are split on whitespace and/or commas
        (honoring simple use of single or double quotes).  If None, then
        no tokenization is performed.

    """

    def __init__(self, itemtype, domain=None, string_lexer=NOTSET):
        self.itemtype = itemtype
        if domain is None:
            self.domain = self.itemtype
        else:
            self.domain = domain
        if string_lexer is NOTSET:
            self.string_lexer = _default_string_list_lexer
        else:
            self.string_lexer = string_lexer
        self.__name__ = 'ListOf(%s)' % (getattr(self.domain, '__name__', self.domain),)

    def __call__(self, value):
        if isinstance(value, str) and self.string_lexer is not None:
            return [self.domain(v) for v in self.string_lexer(value)]
        if hasattr(value, '__iter__') and not isinstance(value, self.itemtype):
            return [self.domain(v) for v in value]
        return [self.domain(value)]

    def domain_name(self):
        _dn = _domain_name(self.domain) or ""
        return f'ListOf[{_dn}]'


class Module(object):
    """Domain validator for modules.

    Modules can be specified as module objects, by module name,
    or by the path to the module's file. If specified by path, the
    path string has the same path expansion features supported by
    the :py:class:`Path` class.

    Note that modules imported by file path may not be recognized as
    part of a package, and as such they should not use relative package
    importing (such as ``from . import foo``).

    Parameters
    ----------
    basePath : None, str, ConfigValue
        The base path that will be prepended to any non-absolute path
        values provided.  If None, defaults to :py:attr:`Path.BasePath`.

    expandPath : bool
        If True, then the value will be expanded and normalized.  If
        False, the string representation of the value will be used
        unchanged.  If None, expandPath will defer to the (negated)
        value of :py:attr:`Path.SuppressPathExpansion`.

    Examples
    --------

    The following code shows the three ways you can specify a module: by file
    name, by module name, or by module object. Regardless of how the module is
    specified, what is stored in the configuration is a module object.

    .. testcode::

        from pyomo.common.config import (
            ConfigDict, ConfigValue, Module
        )
        config = ConfigDict()
        config.declare('my_module', ConfigValue(
            domain=Module(),
        ))
        # Set using file path
        config.my_module = '../../pyomo/common/tests/config_plugin.py'
        # Set using python module name, as a string
        config.my_module = 'os.path'
        # Set using an imported module object
        import os.path
        config.my_module = os.path

    """

    def __init__(self, basePath=None, expandPath=None):
        self.basePath = basePath
        self.expandPath = expandPath

    def __call__(self, module_id):
        # If it's already a module, just return it
        if inspect.ismodule(module_id):
            return module_id

        # Try to import it as a module
        try:
            return importlib.import_module(str(module_id))
        except (ModuleNotFoundError, TypeError):
            # This wasn't a module name
            # Ignore the exception and move on to path-based loading
            pass
        # Any other kind of exception will be thrown out of this method

        # If we're still here, try loading by path
        path_domain = Path(self.basePath, self.expandPath)
        path = path_domain(str(module_id))
        return import_file(path)


class Path(object):
    """
    Domain validator for a
    :py:term:`path-like object <path-like object>`.

    This will admit a path-like object
    and get the object's file system representation
    through :py:obj:`os.fsdecode`.
    It will then
    expand any environment variables and leading usernames (e.g.,
    "~myuser" or "~/") appearing in either the value or the base path
    before concatenating the base path and value, expanding the path to
    an absolute path, and normalizing the path.

    Parameters
    ----------
    basePath: None, str, ConfigValue
        The base path that will be prepended to any non-absolute path
        values provided.  If None, defaults to :py:attr:`Path.BasePath`.

    expandPath: bool
        If True, then the value will be expanded and normalized.  If
        False, the string representation of the value will be returned
        unchanged.  If None, expandPath will defer to the (negated)
        value of :py:attr:`Path.SuppressPathExpansion`

    """

    BasePath = None
    SuppressPathExpansion = False

    def __init__(self, basePath=None, expandPath=None):
        self.basePath = basePath
        self.expandPath = expandPath

    def __call__(self, path):
        path = os.fsdecode(path)
        _expand = self.expandPath
        if _expand is None:
            _expand = not Path.SuppressPathExpansion
        if not _expand:
            return path

        if self.basePath:
            base = self.basePath
        else:
            base = Path.BasePath
        if isinstance(base, ConfigValue):
            base = base.value()
        if base is None:
            base = ""
        else:
            base = str(base).lstrip()

        # We want to handle the CWD variable ourselves.  It should
        # always be in a known location (the beginning of the string)
        if base and base[:6].lower() == '${cwd}':
            base = os.getcwd() + base[6:]
        if path and path[:6].lower() == '${cwd}':
            path = os.getcwd() + path[6:]

        ans = os.path.normpath(
            os.path.abspath(
                os.path.join(
                    os.path.expanduser(os.path.expandvars(base)),
                    os.path.expanduser(os.path.expandvars(path)),
                )
            )
        )
        return ans

    def domain_name(self):
        return _domain_name(type(self))


class PathList(Path):
    """
    Domain validator for a list of
    :py:term:`path-like objects <path-like object>`.

    This admits a path-like object or iterable of such.
    If a path-like object is passed, then
    a singleton list containing the object normalized through
    :py:class:`Path` is returned.
    An iterable of path-like objects is cast to a list, each
    entry of which is normalized through :py:class:`Path`.

    Parameters
    ----------
    basePath: Union[None, str, ConfigValue]
        The base path that will be prepended to any non-absolute path
        values provided.  If None, defaults to :py:attr:`Path.BasePath`.

    expandPath: bool
        If True, then the value will be expanded and normalized.  If
        False, the string representation of the value will be returned
        unchanged.  If None, expandPath will defer to the (negated)
        value of :py:attr:`Path.SuppressPathExpansion`

    """

    def __call__(self, data):
        is_path_like = isinstance(data, (str, bytes)) or hasattr(data, "__fspath__")
        if hasattr(data, "__iter__") and not is_path_like:
            return [super(PathList, self).__call__(i) for i in data]
        else:
            return [super(PathList, self).__call__(data)]


class DynamicImplicitDomain(object):
    """Implicit domain that can return a custom domain based on the key.

    This provides a mechanism for managing plugin-like systems, where
    the key specifies a source for additional configuration information.
    For example, given the plugin module,
    ``pyomo/common/tests/config_plugin.py``:

    .. literalinclude:: /../../pyomo/common/tests/config_plugin.py
       :start-at: import

    .. doctest::
       :hide:

       >>> import importlib
       >>> import pyomo.common.fileutils
       >>> from pyomo.common.config import ConfigDict, DynamicImplicitDomain

    Then we can declare a `:class:``ConfigDict`` that imports the domain
    for specific keys from a module that matches the key name:

    .. testcode::

       def _pluginImporter(name, config):
           mod = importlib.import_module(name)
           return mod.get_configuration(config)
       config = ConfigDict()
       config.declare('plugins', ConfigDict(
           implicit=True,
           implicit_domain=DynamicImplicitDomain(_pluginImporter)))
       config.plugins['pyomo.common.tests.config_plugin'] = {'key1': 5}
       config.display()


    .. testoutput::

       plugins:
         pyomo.common.tests.config_plugin:
           key1: 5
           key2: '5'

    .. note::

       This initializer is only useful for the :py:class:`ConfigDict`
       ``implicit_domain`` argument (and not for "regular" ``domain``
       arguments)

    Parameters
    ----------
    callback: Callable[[str, object], ConfigBase]
        A callable (function) that is passed the ConfigDict key and
        value, and is expected to return the appropriate Config object
        (ConfigValue, ConfigList, or ConfigDict)

    """

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, key, value):
        return self.callback(key, value)


# Note: Enum uses a metaclass to work its magic.  To get a deprecation
# warning when creating a subclass of ConfigEnum, we need to decorate
# the __new__ method here (doing the normal trick of letting the class
# decorator automatically wrap the class __new__ or __init__ methods
# does not behave the way one would think because those methods are
# actually created by the metaclass).  The "empty" class "@deprecated()"
# here will look into the resulting class and extract the docstring from
# the original __new__ to generate the class docstring.
@deprecated()
class ConfigEnum(enum.Enum):
    @deprecated(
        "The ConfigEnum base class is deprecated.  "
        "Directly inherit from enum.Enum and then use "
        "In() or InEnum() as the ConfigValue 'domain' for "
        "validation and int/string type conversions.",
        version='6.0',
    )
    def __new__(cls, value, *args):
        member = object.__new__(cls)
        member._value_ = value
        member._args = args
        return member

    @classmethod
    def from_enum_or_string(cls, arg):
        if type(arg) is str:
            return cls[arg]
        else:
            # Handles enum or integer inputs
            return cls(arg)


def _dump(*args, **kwds):
    # TODO: Change the default behavior to no longer be YAML.
    # This was a legacy decision that may no longer be the best
    # decision, given changes to technology over the years.
    try:
        from yaml import safe_dump as dump
    except ImportError:
        # dump = lambda x,**y: str(x)
        # YAML uses lowercase True/False
        def dump(x, **args):
            if type(x) is bool:
                return str(x).lower()
            if type(x) is type:
                return str(type(x))
            return str(x)

    assert '_dump' in globals()
    globals()['_dump'] = dump
    return dump(*args, **kwds)


def _munge_name(name, space_to_dash=True):
    if space_to_dash:
        name = re.sub(r'\s', '-', name)
    name = re.sub(r'_', '-', name)
    return re.sub(r'[^a-zA-Z0-9-_]', '_', name)


def _domain_name(domain):
    if domain is None:
        return ""
    if isinstance(domain, str):
        return domain
    if hasattr(domain, 'domain_name') and not isinstance(domain, type):
        dn = domain.domain_name
        if hasattr(dn, '__call__'):
            return dn()
        else:
            return dn
    if domain.__module__ == 'builtins':
        module = ""
    else:
        module = "~" + domain.__module__ + '.'
    if isinstance(domain, type):
        if building_documentation():
            return module + domain.__qualname__
        else:
            return domain.__name__
    elif inspect.isfunction(domain):
        if building_documentation():
            return module + domain.__qualname__
        else:
            return domain.__name__
    else:
        return None


_leadingSpace = re.compile('^([ \t]*)')


def _strip_indentation(doc):
    if not doc:
        return doc
    lines = doc.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) == 1:
        return doc.lstrip()
    minIndent = min(len(_leadingSpace.match(l).group(0)) for l in lines[1:])
    if len(_leadingSpace.match(lines[0]).group(0)) <= minIndent:
        lines[0] = lines[0].strip()
    else:
        lines[0] = lines[0][minIndent:].rstrip()
    for i, l in enumerate(lines[1:]):
        lines[i + 1] = l[minIndent:].rstrip()
    return '\n'.join(lines)


def _value2string(prefix, value, obj):
    _str = prefix
    if value is not None:
        try:
            data = value.value(False) if value is obj else value
            if getattr(builtins, data.__class__.__name__, None) is not None:
                _str += _dump(
                    data, default_flow_style=True, allow_unicode=True
                ).rstrip()
                if _str.endswith("..."):
                    _str = _str[:-3].rstrip()
            else:
                _str += str(data)
        except:
            _str += str(type(data))
    return _str.rstrip()


def _value2yaml(prefix, value, obj):
    _str = prefix
    if value is not None:
        try:
            data = value.value(False) if value is obj else value
            _str += _dump(data, default_flow_style=True).rstrip()
            if _str.endswith("..."):
                _str = _str[:-3].rstrip()
        except:
            _str += str(type(data))
    return _str.rstrip()


class _UnpickleableDomain(object):
    def __init__(self, obj):
        self._type = type(obj).__name__
        self._name = obj.name(True)

    def __call__(self, arg):
        logger.error(
            """%s '%s' was pickled with an unpicklable domain.
    The domain was stripped and lost during the pickle process.  Setting
    new values on the restored object cannot be mapped into the correct
    domain.
"""
            % (self._type, self._name)
        )
        return arg


def _picklable(field, obj):
    ftype = type(field)
    # If the field is a type (class, etc), cache the 'known' status of
    # the actual field type and not the generic 'type' class
    if ftype is type:
        ftype = field
    if ftype in _picklable.known:
        return field if _picklable.known[ftype] else _UnpickleableDomain(obj)
    try:
        pickle.dumps(field)
        if ftype not in _picklable.unknowable_types:
            _picklable.known[ftype] = True
        return field
    except:
        # Contrary to the documentation, Python is not at all consistent
        # with the exception that is raised when pickling an object
        # fails:
        #
        #    Python 2.6 - 3.4:  pickle.PicklingError
        #    Python 3.5 - 3.6:  AttributeError
        #    Python 2.6 - 2.7 (cPickle):  TypeError
        #
        # What we are concerned about is masking things like recursion
        # errors.  Unfortunately, Python is not quite consistent there,
        # either: exceeding recursion depth raises a RuntimeError
        # through 3.4, then switches to a RecursionError (a derivative
        # of RuntimeError).
        if isinstance(sys.exc_info()[0], RuntimeError):
            raise
        if ftype not in _picklable.unknowable_types:
            _picklable.known[ftype] = False
        return _UnpickleableDomain(obj)


_picklable.known = {}
# The "picklability" of some types is not categorically "knowable"
# (e.g., functions can be pickled, but only if they are declared at the
# module scope)
_picklable.unknowable_types = {type, types.FunctionType}

_store_bool = {'store_true', 'store_false'}


def _build_lexer(literals=''):
    # Ignore whitespace (space, tab, linefeed, and comma)
    t_ignore = " \t\r,"

    tokens = ["STRING", "WORD"]  # quoted string  # unquoted string

    # A "string" is a proper quoted string
    _quoted_str = r"'(?:[^'\\]|\\.)*'"
    _general_str = "|".join([_quoted_str, _quoted_str.replace("'", '"')])

    @ply.lex.TOKEN(_general_str)
    def t_STRING(t):
        t.value = t.value[1:-1]
        return t

    # A "word" contains no whitesspace or commas
    @ply.lex.TOKEN(r'[^' + repr(t_ignore + literals) + r']+')
    def t_WORD(t):
        t.value = t.value
        return t

    # Error handling rule
    def t_error(t):
        # Note this parser does not allow "\n", so lexpos is the
        # column number
        raise IOError(
            "ERROR: Token '%s' Line %s Column %s" % (t.value, t.lineno, t.lexpos + 1)
        )

    return ply.lex.lex()


def _default_string_list_lexer(value):
    """Simple string tokenizer for lists of words.

    This default lexer splits strings on whitespace and/or commas while
    honoring use of single and double quotes.  Separators (whitespace or
    commas) are not returned.  Consecutive delimiters are ignored (and
    do not yield empty strings).

    """
    _lex = _default_string_list_lexer._lex
    if _lex is None:
        _default_string_list_lexer._lex = _lex = _build_lexer()
    _lex.input(value)
    while True:
        tok = _lex.token()
        if not tok:
            break
        yield tok.value


_default_string_list_lexer._lex = None


def _default_string_dict_lexer(value):
    """Simple string tokenizer for dict data.

    This default lexer splits strings on whitespace and/or commas while
    honoring use of single and double quotes.  ':' and '=' are
    recognized as special tokens.  Separators (whitespace or commas) are
    not returned.  Consecutive delimiters are ignored (and do not yield
    empty strings).

    """
    _lex = _default_string_dict_lexer._lex
    if _lex is None:
        _default_string_dict_lexer._lex = _lex = _build_lexer(':=')
    _lex.input(value)
    while True:
        key = _lex.token()
        if not key:
            break
        sep = _lex.token()
        if not sep:
            raise ValueError("Expected ':' or '=' but encountered end of string")
        if sep.type not in ':=':
            raise ValueError(
                f"Expected ':' or '=' but found '{sep.value}' at "
                f"Line {sep.lineno} Column {sep.lexpos+1}"
            )
        val = _lex.token()
        if not val:
            raise ValueError(
                f"Expected value following '{sep.type}' "
                f"but encountered end of string"
            )
        yield key.value, val.value


_default_string_dict_lexer._lex = None


def _formatter_str_to_callback(pattern, formatter):
    "Wrapper function that converts formatter strings to callback functions"

    if not pattern:
        pattern = ''
    if '%s' in pattern:
        cb = lambda self, indent, obj: self.out.write(indent + pattern % obj.name())
    elif pattern:
        cb = lambda self, indent, obj: self.out.write(indent + pattern)
    else:
        cb = lambda self, indent, obj: None
    return types.MethodType(cb, formatter)


def _formatter_str_to_item_callback(pattern, formatter):
    "Wrapper function that converts item formatter strings to callback functions"

    if not pattern:
        pattern = ''
    if '%s' in pattern:
        _item_body_formatter = lambda doc: pattern % (doc,)
    else:
        _item_body_formatter = lambda doc: pattern

    def _item_body_cb(self, indent, obj):
        _doc = obj._doc or obj._description or ""
        if not _doc:
            return ''
        wraplines = '\n ' not in _doc
        _doc = _item_body_formatter(_doc).rstrip()
        if not _doc:
            return ''
        _indent = indent + ' ' * self.indent_spacing
        if wraplines:
            doc_lines = textwrap.wrap(
                _doc, self.width, initial_indent=_indent, subsequent_indent=_indent
            )
            self.out.write(('\n'.join(doc_lines)).rstrip() + '\n')
        elif _doc.lstrip() == _doc:
            self.out.write(_indent + _doc + '\n')
        else:
            self.out.write(_doc + '\n')

    return types.MethodType(_item_body_cb, formatter)


class ConfigFormatter(object):
    def _initialize(self, indent_spacing, width, visibility):
        self.out = io.StringIO()
        self.indent_spacing = indent_spacing
        self.width = width
        self.visibility = visibility

    def _block_start(self, indent, obj):
        pass

    def _block_end(self, indent, obj):
        pass

    def _item_start(self, indent, obj):
        pass

    def _item_body(self, indent, obj):
        pass

    def _item_end(self, indent, obj):
        pass

    def _finalize(self):
        return self.out.getvalue()

    def generate(self, config, indent_spacing=2, width=78, visibility=None):
        self._initialize(indent_spacing, width, visibility)
        self._level = level = []
        lastObj = config
        indent = ''
        for lvl, pre, val, obj in config._data_collector(1, '', visibility, True):
            if len(level) < lvl:
                while len(level) < lvl - 1:
                    level.append(None)
                level.append(lastObj)
                self._block_start(indent, lastObj)
                indent += ' ' * indent_spacing
            while len(level) > lvl:
                _last = level.pop()
                if _last is not None:
                    indent = indent[:-indent_spacing]
                    self._block_end(indent, _last)

            lastObj = obj
            self._item_start(indent, obj)
            self._item_body(indent, obj)
            self._item_end(indent, obj)
        while level:
            _last = level.pop()
            if _last is not None:
                indent = indent[:-indent_spacing]
                self._block_end(indent, _last)
        return self._finalize()


class String_ConfigFormatter(ConfigFormatter):
    def __init__(self, block_start, block_end, item_start, item_body, item_end):
        self._block_start = _formatter_str_to_callback(block_start, self)
        self._block_end = _formatter_str_to_callback(block_end, self)
        self._item_start = _formatter_str_to_callback(item_start, self)
        self._item_end = _formatter_str_to_callback(item_end, self)
        self._item_body = _formatter_str_to_item_callback(item_body, self)


class LaTeX_ConfigFormatter(String_ConfigFormatter):
    def __init__(self):
        super().__init__(
            "\\begin{description}[topsep=0pt,parsep=0.5em,itemsep=-0.4em]\n",
            "\\end{description}\n",
            "\\item[{%s}]\\hfill\n",
            "\\\\%s",
            "",
        )


class numpydoc_ConfigFormatter(ConfigFormatter):
    def _initialize(self, *args):
        super()._initialize(*args)
        self.wrapper = textwrap.TextWrapper(width=self.width)
        self.suppress = None

    def _block_start(self, indent, obj):
        if len(self._level) > 1 and obj.domain_name() != "dict":
            self.suppress = obj

    def _block_end(self, indent, obj):
        if self.suppress is obj:
            self.suppress = None

    def _item_body(self, indent, obj):
        if self.suppress is not None:
            return
        typeinfo = ', '.join(
            filter(
                None,
                [
                    obj.domain_name(),
                    (
                        'optional'
                        if obj._default is None
                        else f'default={obj._default!r}'
                    ),
                ],
            )
        )
        # Note that numpydoc / ReST specifies that the colon in
        # definition lists be surrounded by spaces (i.e., " : ").
        # However, as of numpydoc (1.1.0) / Sphinx (3.4.3) / napoleon
        # (0.7), things aren't really geared for nested lists of
        # parameters.  Definition lists omit the colon, and
        # sub-definitions are rendered as normal definition sections
        # (without the special formatting applied to Parameters lists),
        # leading to less readable docs.  As they tolerate omitting the
        # space before the colon at the top level (which at lower levels
        # causes nested definition lists to NOT omit the colon), we will
        # generate non-standard ReST and omit the preceding space:
        self.out.write(f'\n{indent}{obj.name()}: {typeinfo}\n')
        self.wrapper.initial_indent = indent + ' ' * self.indent_spacing
        self.wrapper.subsequent_indent = indent + ' ' * self.indent_spacing
        vis = ""
        if self.visibility is None and obj._visibility >= ADVANCED_OPTION:
            vis = "[ADVANCED option]"
            if obj._visibility >= DEVELOPER_OPTION:
                vis = "[DEVELOPER option]"
        itemdoc = wrap_reStructuredText(
            '\n\n'.join(
                filter(
                    None, [vis, inspect.cleandoc(obj._doc or obj._description or "")]
                )
            ),
            self.wrapper,
        )
        if itemdoc:
            self.out.write('\n' + itemdoc + '\n')

    def _finalize(self):
        return inspect.cleandoc(self.out.getvalue())


ConfigFormatter.formats = {
    'latex': LaTeX_ConfigFormatter,
    'numpydoc': numpydoc_ConfigFormatter,
}


@deprecated(
    "add_docstring_list is deprecated.  Please use the "
    "@document_kwargs_from_configdict() decorator.",
    version='6.6.0',
)
def add_docstring_list(docstring, configdict, indent_by=4):
    """Returns the docstring with a formatted configuration arguments listing."""
    section = 'Keyword Arguments'
    return (
        inspect.cleandoc(docstring)
        + '\n'
        + section
        + '\n'
        + '-' * len(section)
        + '\n'
        + configdict.generate_documentation(
            indent_spacing=indent_by, width=256, visibility=0, format='numpydoc'
        )
    )


class document_kwargs_from_configdict(object):
    """Decorator to append the documentation of a ConfigDict to the docstring

    This adds the documentation of the specified :py:class:`ConfigDict`
    (using the :py:class:`numpydoc_ConfigFormatter` formatter) to the
    decorated object's docstring.

    Parameters
    ----------
    config : ConfigDict or str
        the :py:class:`ConfigDict` to document.  If a ``str``, then the
        :py:class:`ConfigDict` is obtained by retrieving the named
        attribute from the decorated object (thereby enabling
        documenting class objects whose ``__init__`` keyword arguments
        are processed by a :py:class:`ConfigDict` class attribute)

    section : str
        the section header to preface config documentation with

    indent_spacing : int
        number of spaces to indent each block of documentation

    width : int
        total documentation width in characters (for wrapping paragraphs)

    doc : str, optional
        the initial docstring to append the ConfigDict documentation to.
        If None, then the decorated object's ``__doc__`` will be used.

    Examples
    --------

    >>> from pyomo.common.config import (
    ...     ConfigDict, ConfigValue, document_kwargs_from_configdict
    ... )
    >>> class MyClass(object):
    ...     CONFIG = ConfigDict()
    ...     CONFIG.declare('iterlim', ConfigValue(
    ...         default=3000,
    ...         domain=int,
    ...         doc="Iteration limit.  Specify None for no limit"
    ...     ))
    ...     CONFIG.declare('tee', ConfigValue(
    ...         domain=bool,
    ...         doc="If True, stream the solver output to the console"
    ...     ))
    ...
    ...     @document_kwargs_from_configdict(CONFIG)
    ...     def solve(self, **kwargs):
    ...         config = self.CONFIG(kwargs)
    ...         # ...
    ...
    >>> help(MyClass.solve)
    Help on function solve:
    <BLANKLINE>
    solve(self, **kwargs)
        Keyword Arguments
        -----------------
        iterlim: int, default=3000
            Iteration limit.  Specify None for no limit
    <BLANKLINE>
        tee: bool, optional
            If True, stream the solver output to the console

    """

    def __init__(
        self,
        config,
        section='Keyword Arguments',
        indent_spacing=4,
        width=78,
        visibility=None,
        doc=None,
        preamble=None,
    ):
        self.config = config
        self.section = section
        self.indent_spacing = indent_spacing
        self.width = width
        self.visibility = visibility
        self.doc = doc
        self.preamble = preamble

    def __call__(self, fcn):
        config = self.config
        section = self.section
        if isinstance(config, str):
            config = getattr(fcn, config)
        if section and '\n' not in section:
            section += '\n' + '-' * len(section) + '\n'
        if self.doc is not None:
            doc = inspect.cleandoc(self.doc)
        elif fcn.__doc__:
            doc = inspect.cleandoc(fcn.__doc__)
        else:
            doc = ""
        doc = self._ensure_blank_line(doc)
        if self.preamble:
            doc += self._ensure_blank_line(inspect.cleandoc(self.preamble))
        fcn.__doc__ = (
            doc
            + f'{section}'
            + config.generate_documentation(
                indent_spacing=self.indent_spacing,
                width=self.width,
                visibility=self.visibility,
                format='numpydoc',
            )
        )
        return fcn

    def _ensure_blank_line(self, val):
        if not val:
            return val
        if not val.endswith('\n'):
            return val + '\n\n'
        return val + '\n'


def _method_wrapper(func):
    """A simplified :class:`functools.partialmethod` that duplicates the __doc__

    Ideally we could use :class:`functools.partialmethod` to copy a
    method from a base class to a derived class so that we can set a
    docstring that is customized for the derived class.  Unfortunately,
    :class:`functools.partialmethod` returns a method descriptor.  Here
    we will declare a new method (with a generic signature) and then
    leverage :func:`functools.wraps` to copy over the original signature
    and docstring.

    """

    @functools.wraps(func)
    def method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return method


class document_configdict(document_kwargs_from_configdict):
    """A simplified wrapper around :class:`document_kwargs_from_configdict`
    for documenting classes derived from ConfigDict.

    """

    def __init__(
        self,
        section='Options',
        indent_spacing=4,
        width=78,
        visibility=None,
        doc=None,
        preamble=None,
    ):
        return super().__init__(
            config=None,
            section=section,
            indent_spacing=indent_spacing,
            width=width,
            visibility=visibility,
            doc=doc,
            preamble=preamble,
        )

    def __call__(self, fcn):
        self.config = fcn()
        return super().__call__(fcn)


class document_class_CONFIG(document_kwargs_from_configdict):
    def __init__(
        self,
        section='CONFIG',
        indent_spacing=4,
        width=78,
        visibility=None,
        doc=None,
        preamble=None,
        methods=None,
    ):
        self.methods = methods
        return super().__init__(
            config=None,
            section=section,
            indent_spacing=indent_spacing,
            width=width,
            visibility=visibility,
            doc=doc,
            preamble=preamble,
        )

    def __call__(self, cls):
        self.config = cls.CONFIG
        if cls.__module__ == 'builtins':
            # Suppress "builtins" module (appears as part of
            # documentation tests)
            module = ""
        else:
            module = cls.__module__ + '.'
        ref = f"{module}{cls.__name__}::CONFIG"
        self.config.set_domain(f":ref:`{cls.__name__}.CONFIG <{ref}>`")
        if self.preamble is None:
            self.preamble = f"""**Class configuration**

This class leverages the Pyomo Configuration System for managing
configuration options.  See the discussion on :ref:`configuring class
hierarchies <class_config>` for more information on how configuration
class attributes, instance attributes, and method keyword arguments
interact."""

        self.preamble += f"\n\n.. _{ref}:\n"
        if self.methods:
            for method in self.methods:
                if method not in vars(cls):
                    # If this method is inherited, we need to make a
                    # "local" version of it so we don't change the
                    # docstring on the base class.
                    setattr(cls, method, _method_wrapper(getattr(cls, method)))
                document_kwargs_from_configdict(
                    self.config,
                    indent_spacing=self.indent_spacing,
                    width=self.width,
                    visibility=self.visibility,
                    doc=self.doc,
                )(getattr(cls, method))
        return super().__call__(cls)


class UninitializedMixin(object):
    """Mixin class to support delayed data initialization.

    This mixin can be used to create a derived Config class that hides
    the (uninitialized) ``_data`` attribute behind a property.  Any
    attempt to access the ``_data`` will trigger the initialization of the
    Config object from its ``_default`` value.  Setting the ``_data``
    attribute will also trigger resolution of the Config object, but
    without processing the ``_default``.

    """

    __slots__ = ()

    @property
    def _data(self):
        #
        # We assume that _default is usually a concrete value.  But, we
        # also accept a types (classes) and initialization functions as
        # defaults, in which case we will construct an instance of that
        # class and use that as the default.  If they both raise
        # exceptions, we will let the original exception propagate up.
        #
        try:
            self._setter(self._default)
        except:
            if hasattr(self._default, '__call__'):
                _default_val = self._default()
                try:
                    self._setter(_default_val)
                    return self._data
                except:
                    pass
            raise
        return self._data

    @_data.setter
    def _data(self, value):
        _mro = self.__class__.__mro__
        # There is an edge case in multithreaded environments where this
        # function could actually be called more than once for a single
        # ConfigValue.  We want to make sure that only the first of the
        # calls actually updates the __class__ (the others will
        # recursively lookup the _data attribute and the second lookup
        # will resolve to normal attribute assignment).
        #
        # We first encountered this issue for Config objects stored as
        # class attributes (i.e., the default Config for something like
        # a solver or writer) and multiple threads were simultaneously
        # creating instances of the class (each of which was resolving
        # the default values for the class attribute).
        #
        # Note that this explicitly assumes that the uninitialized
        # Config object was defined as:
        #
        #    class UninitializedConfig(UninitializedMixin, Config)
        #
        # and that the resulting class was never inherited from.  If
        # this assumption is ever violated, attempts to use the
        # uninitialized config object will generate infinite recursion
        # (and that is OK, as the developer should immediately be
        # informed of their error)
        if _mro[1] is UninitializedMixin:
            self.__class__ = _mro[2]
        self._data = value


class ConfigBase(object):
    # Note: __getstate__ relies on this field ordering.  Do not change.
    __slots__ = (
        '_parent',
        '_domain',
        '_name',
        '_userSet',
        '_userAccessed',
        '_data',
        '_default',
        '_description',
        '_doc',
        '_visibility',
        '_argparse',
    )

    # This just needs to be any singleton-like object; we use it so that
    # we can tell if an argument is provided (and we can't use None as
    # None is a valid user-specified argument).  Making it a class helps
    # when Config objects are pickled.
    class NoArgument(object):
        pass

    def __init__(
        self, default=None, domain=None, description=None, doc=None, visibility=0
    ):
        self._parent = None
        self._name = None
        self._userSet = False
        self._userAccessed = False

        self._data = NOTSET
        self._default = default
        self._domain = domain
        self._description = _strip_indentation(description)
        self._doc = _strip_indentation(doc)
        self._visibility = visibility
        self._argparse = None
        if self._UninitializedClass is not None:
            self.__class__ = self._UninitializedClass

    def __getstate__(self):
        # Nominally, __getstate__() should return:
        #
        # state = super(Class, self).__getstate__()
        # for i in Class.__slots__:
        #    state[i] = getattr(self,i)
        # return state
        #
        # However, in this case, the (nominal) parent class is
        # 'object', and object does not implement __getstate__.  Since
        # super() doesn't actually return a class, we are going to check
        # the *derived class*'s MRO and see if this is the second to
        # last class (the last is always 'object').  If it is, then we
        # can allocate the state dictionary.  If it is not, then we call
        # the super-class's __getstate__ (since that class is NOT
        # 'object').
        state = [None, _picklable(self._domain, self)]
        # Note: [2:] skips _parent and _domain (intentionally): We just
        # wrapped _domain in _picklable and explicitly set _parent to
        # None (it will be restored in __setstate__).
        state.extend(getattr(self, key) for key in ConfigBase.__slots__[2:])
        return state

    def __setstate__(self, state):
        for key, val in zip(ConfigBase.__slots__, state):
            # Note: per the Python data model docs, we explicitly
            # set the attribute using object.__setattr__() instead
            # of setting self.__dict__[key] = val.
            object.__setattr__(self, key, val)

    def __call__(
        self,
        value=NOTSET,
        default=NOTSET,
        domain=NOTSET,
        description=NOTSET,
        doc=NOTSET,
        visibility=NOTSET,
        implicit=NOTSET,
        implicit_domain=NOTSET,
        preserve_implicit=False,
    ):
        # We will pass through overriding arguments to the constructor.
        # This way if the constructor does special processing of any of
        # the arguments (like implicit_domain), we don't have to repeat
        # that code here.  Unfortunately, it means we need to do a bit
        # of logic to be sure we only pass through appropriate
        # arguments.
        kwds = {}
        fields = ('description', 'doc', 'visibility')
        if isinstance(self, ConfigDict):
            fields += (('implicit', '_implicit_declaration'), 'implicit_domain')
            assert domain is NOTSET
            assert default is NOTSET
        else:
            fields += ('domain',)
            if default is NOTSET:
                default = self.value()
                if default is NOTSET:
                    default = None
            kwds['default'] = default
            assert implicit is NOTSET
            assert implicit_domain is NOTSET
        for field in fields:
            if type(field) is tuple:
                field, attr = field
            else:
                attr = '_' + field
            if locals()[field] is NOTSET:
                kwds[field] = getattr(self, attr, NOTSET)
            else:
                kwds[field] = locals()[field]

        # Initialize the new config object
        ans = self.__class__(**kwds)

        if isinstance(self, ConfigDict):
            # Copy over any Dict definitions
            ans._domain = self._domain
            for k, v in self._data.items():
                if preserve_implicit or k in self._declared:
                    ans._data[k] = _tmp = v(preserve_implicit=preserve_implicit)
                    if k in self._declared:
                        ans._declared.add(k)
                    _tmp._parent = ans
                    _tmp._name = v._name

        # ... and set the value, if appropriate
        if value is not NOTSET:
            ans.set_value(value)
        return ans

    def name(self, fully_qualified=False):
        # Special case for the top-level dict
        if self._name is None:
            return ""
        elif fully_qualified and self._parent is not None:
            pName = self._parent.name(fully_qualified)
            # Special case for ConfigList indexing and the top-level entries
            if self._name.startswith('[') or not pName:
                return pName + self._name
            else:
                return pName + '.' + self._name
        else:
            return self._name

    def domain_name(self):
        _dn = _domain_name(self._domain)
        if _dn is None:
            return _munge_name(self.name(), False)
        return _dn

    def set_default_value(self, default):
        self._default = default

    def set_domain(self, domain):
        self._domain = domain
        # Note that the domain is generally a callable (type, function,
        # or functor).  However, ConfigDict can also have a str domain
        # (because ConfigDict doesn't need/use an actual domain for
        # validation, we re-use that slot to *document* the domain).
        if domain.__class__ is not str:
            self.set_value(self.value(accessValue=False))

    def _cast(self, value):
        if value is None:
            return value
        if self._domain is not None:
            try:
                if value is not NOTSET:
                    return self._domain(value)
                else:
                    return self._domain()
            except:
                err = sys.exc_info()[1]
                if hasattr(self._domain, '__name__'):
                    _dom = self._domain.__name__
                else:
                    _dom = type(self._domain)
                raise ValueError(
                    "invalid value for configuration '%s':\n"
                    "\tFailed casting %s\n\tto %s\n\tError: %s"
                    % (self.name(True), value, _dom, err)
                )
        else:
            return value

    def reset(self):
        # Reset the object back to its default value and clear the
        # userSet and userAccessed flags
        self._UninitializedClass._data.fget(self)
        self._userAccessed = False
        self._userSet = False

    def declare_as_argument(self, *args, **kwds):
        """Map this Config item to an argparse argument.

        Valid arguments include all valid arguments to argparse's
        ArgumentParser.add_argument() with the exception of 'default'.
        In addition, you may provide a group keyword argument to either
        pass in a pre-defined option group or subparser, or else pass in
        the string name of a group, subparser, or (subparser, group).

        """

        if 'default' in kwds:
            raise TypeError(
                "You cannot specify an argparse default value with "
                "ConfigBase.declare_as_argument().  The default value is "
                "supplied automatically from the Config definition."
            )

        if 'action' not in kwds and self._domain is bool:
            if not self._default:
                kwds['action'] = 'store_true'
            else:
                kwds['action'] = 'store_false'
                if not args:
                    args = ('--disable-' + _munge_name(self.name()),)
                if 'help' not in kwds:
                    kwds['help'] = "[DON'T] " + self._description
        if 'help' not in kwds:
            kwds['help'] = self._description
        if not args:
            args = ('--' + _munge_name(self.name()),)
        if self._argparse:
            self._argparse = self._argparse + ((args, kwds),)
        else:
            self._argparse = ((args, kwds),)
        return self

    def initialize_argparse(self, parser):
        def _get_subparser_or_group(_parser, name):
            # Note: strings also have a 'title()' method.  We are
            # looking for things that look like argparse
            # groups/subparsers, so just checking for the attribute
            # is insufficient: it needs to be a string attribute as
            # well
            if isinstance(name, argparse._ActionsContainer):
                # hasattr(_group, 'title') and \
                #    isinstance(_group.title, str):
                return 2, name

            if not isinstance(name, str):
                raise RuntimeError(
                    'Unknown datatype (%s) for argparse group on '
                    'configuration definition %s'
                    % (type(name).__name__, obj.name(True))
                )

            try:
                for _grp in _parser._subparsers._group_actions:
                    if name in _grp._name_parser_map:
                        return 1, _grp._name_parser_map[name]
            except AttributeError:
                pass

            for _grp in _parser._action_groups:
                if _grp.title == name:
                    return 0, _grp
            return 0, _parser.add_argument_group(title=name)

        def _process_argparse_def(obj, _args, _kwds):
            _parser = parser
            # shallow copy the dict so we can remove the group flag and
            # add things like documentation, etc.
            _kwds = dict(_kwds)
            if 'group' in _kwds:
                _group = _kwds.pop('group')
                if isinstance(_group, tuple):
                    for _idx, _grp in enumerate(_group):
                        _issub, _parser = _get_subparser_or_group(_parser, _grp)
                        if not _issub and _idx < len(_group) - 1:
                            raise RuntimeError(
                                "Could not find argparse subparser '%s' for "
                                "Config item %s" % (_grp, obj.name(True))
                            )
                else:
                    _issub, _parser = _get_subparser_or_group(_parser, _group)
            if 'dest' not in _kwds:
                _kwds['dest'] = 'CONFIGBLOCK.' + obj.name(True)
                if (
                    'metavar' not in _kwds
                    and _kwds.get('action', '') not in _store_bool
                    and obj._domain is not None
                ):
                    _kwds['metavar'] = obj.domain_name().upper()
            _parser.add_argument(*_args, default=argparse.SUPPRESS, **_kwds)

        for level, prefix, value, obj in self._data_collector(None, ""):
            if obj._argparse is None:
                continue
            for _args, _kwds in obj._argparse:
                _process_argparse_def(obj, _args, _kwds)

    def import_argparse(self, parsed_args):
        for level, prefix, value, obj in self._data_collector(None, ""):
            if obj._argparse is None:
                continue
            for _args, _kwds in obj._argparse:
                if 'dest' in _kwds:
                    _dest = _kwds['dest']
                    if _dest in parsed_args:
                        obj.set_value(parsed_args.__dict__[_dest])
                else:
                    _dest = 'CONFIGBLOCK.' + obj.name(True)
                    if _dest in parsed_args:
                        obj.set_value(parsed_args.__dict__[_dest])
                        del parsed_args.__dict__[_dest]
        return parsed_args

    def display(
        self, content_filter=None, indent_spacing=2, ostream=None, visibility=None
    ):
        if content_filter not in ConfigDict.content_filters:
            raise ValueError(
                "unknown content filter '%s'; valid values are %s"
                % (content_filter, ConfigDict.content_filters)
            )
        _blocks = []
        if ostream is None:
            ostream = sys.stdout

        for lvl, prefix, value, obj in self._data_collector(0, "", visibility):
            _str = _value2string(prefix, value, obj)
            _blocks[lvl:] = [' ' * indent_spacing * lvl + _str + "\n"]
            if content_filter == 'userdata' and not obj._userSet:
                continue
            for i, v in enumerate(_blocks):
                if v is not None:
                    ostream.write(v)
                    _blocks[i] = None

    def generate_yaml_template(self, indent_spacing=2, width=78, visibility=0):
        minDocWidth = 20
        comment = "  # "
        data = list(self._data_collector(0, "", visibility))
        level_info = {}
        for lvl, pre, val, obj in data:
            _str = _value2yaml(pre, val, obj)
            if lvl not in level_info:
                level_info[lvl] = {'data': [], 'off': 0, 'line': 0, 'over': 0}
            level_info[lvl]['data'].append(
                (_str.find(':') + 2, len(_str), len(obj._description or ""))
            )
        for lvl in sorted(level_info):
            indent = lvl * indent_spacing
            _ok = width - indent - len(comment) - minDocWidth
            offset = max(
                val if val < _ok else key for key, val, doc in level_info[lvl]['data']
            )
            offset += indent + len(comment)
            over = sum(
                1 for key, val, doc in level_info[lvl]['data'] if doc + offset > width
            )
            if len(level_info[lvl]['data']) - over > 0:
                line = max(
                    offset + doc
                    for key, val, doc in level_info[lvl]['data']
                    if offset + doc <= width
                )
            else:
                line = width
            level_info[lvl]['off'] = offset
            level_info[lvl]['line'] = line
            level_info[lvl]['over'] = over
        maxLvl = 0
        maxDoc = 0
        pad = 0
        for lvl in sorted(level_info):
            _pad = level_info[lvl]['off']
            _doc = level_info[lvl]['line'] - _pad
            if _pad > pad:
                if maxDoc + _pad <= width:
                    pad = _pad
                else:
                    break
            if _doc + pad > width:
                break
            if _doc > maxDoc:
                maxDoc = _doc
            maxLvl = lvl
        os = io.StringIO()
        if self._description:
            os.write(comment.lstrip() + self._description + "\n")
        for lvl, pre, val, obj in data:
            _str = _value2yaml(pre, val, obj)
            if not obj._description:
                os.write(' ' * indent_spacing * lvl + _str + "\n")
                continue
            if lvl <= maxLvl:
                field = pad - len(comment)
            else:
                field = level_info[lvl]['off'] - len(comment)
            os.write(' ' * indent_spacing * lvl)
            if width - len(_str) - minDocWidth >= 0:
                os.write('%%-%ds' % (field - indent_spacing * lvl) % _str)
            else:
                os.write(_str + '\n' + ' ' * field)
            os.write(comment)
            txtArea = max(width - field - len(comment), minDocWidth)
            os.write(
                ("\n" + ' ' * field + comment).join(
                    textwrap.wrap(obj._description, txtArea, subsequent_indent='  ')
                )
            )
            os.write('\n')
        return os.getvalue()

    def generate_documentation(
        self,
        block_start=None,
        block_end=None,
        item_start=None,
        item_body=None,
        item_end=None,
        indent_spacing=2,
        width=78,
        visibility=None,
        format='latex',
    ):
        if isinstance(format, str):
            formatter = ConfigFormatter.formats.get(format, None)
            if formatter is None:
                raise ValueError(f"Unrecognized documentation formatter, '{format}'")
            formatter = formatter()
        else:
            # Assume everything not a str is a valid formatter object.
            formatter = format

        deprecated_args = (block_start, block_end, item_start, item_end)
        if any(arg is not None for arg in deprecated_args):
            names = ('block_start', 'block_end', 'item_start', 'item_end')
            for arg, name in zip(deprecated_args, names):
                if arg is None:
                    continue
                deprecation_warning(
                    f"Overriding '{name}' by passing strings to "
                    "generate_documentation is deprecated.  Create an instance of a "
                    "StringConfigFormatter and pass it as the 'format' argument.",
                    version='6.6.0',
                )
                setattr(
                    formatter, "_" + name, _formatter_str_to_callback(arg, formatter)
                )
        if item_body is not None:
            deprecation_warning(
                "Overriding 'item_body' by passing strings to "
                "generate_documentation is deprecated.  Create an instance of a "
                "StringConfigFormatter and pass it as the 'format' argument.",
                version='6.6.0',
            )
            setattr(
                formatter,
                "_item_body",
                _formatter_str_to_item_callback(item_body, formatter),
            )

        return formatter.generate(self, indent_spacing, width, visibility)

    def user_values(self):
        if self._userSet:
            yield self
        for level, prefix, value, obj in self._data_collector(0, ""):
            if obj._userSet:
                yield obj

    def unused_user_values(self):
        if self._userSet and not self._userAccessed:
            yield self
        for level, prefix, value, obj in self._data_collector(0, ""):
            if obj._userSet and not obj._userAccessed:
                yield obj


class ConfigValue(ConfigBase):
    """Store and manipulate a single configuration value.

    Parameters
    ----------
    default: optional
        The default value that this ConfigValue will take if no value is
        provided.

    domain: Callable, optional
        The domain can be any callable that accepts a candidate value
        and returns the value converted to the desired type, optionally
        performing any data validation.  The result will be stored into
        the ConfigValue.  Examples include type constructors like `int`
        or `float`.  More complex domain examples include callable
        objects; for example, the :py:class:`In` class that ensures that
        the value falls into an acceptable set or even a complete
        :py:class:`ConfigDict` instance.

    description: str, optional
        The short description of this value

    doc: str, optional
        The long documentation string for this value

    visibility: int, optional
        The visibility of this ConfigValue when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigValues with visibility=0 (the
        default) will always be printed / included.  ConfigValues
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    __slots__ = ()

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return self._data

    def _setter(self, value):
        self._data = self._cast(value)

    def set_value(self, value):
        self._setter(value)
        self._userSet = True

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        yield (level, prefix, self, self)


ConfigValue._UninitializedClass = type(
    'UninitializedConfigValue', (UninitializedMixin, ConfigValue), {'__slots__': ()}
)


class ImmutableConfigValue(ConfigValue):
    __slots__ = ()

    def __new__(self, *args, **kwds):
        # ImmutableConfigValue objects are never directly created, and
        # any attempt to copy one will generate a mutable ConfigValue
        # object
        return ConfigValue(*args, **kwds)

    def _setter(self, value):
        try:
            _data = self._data
            super()._setter(value)
            if _data != self._data:
                raise RuntimeError(f"'{self.name(True)}' is currently immutable")
        finally:
            self._data = _data


class MarkImmutable(object):
    """
    Mark instances of ConfigValue as immutable.

    Parameters
    ----------
    config_value: ConfigValue
        The ConfigValue instances that should be marked immutable.
        Note that multiple instances of ConfigValue can be passed.

    Examples
    --------
    .. testcode::

       config = ConfigDict()
       config.declare('a', ConfigValue(default=1, domain=int))
       config.declare('b', ConfigValue(default=1, domain=int))
       locker = MarkImmutable(config.get('a'), config.get('b'))

    Now, config.a and config.b cannot be changed:

    .. doctest::

       >>> config.a = 5
       Traceback (most recent call last):
          ...
       RuntimeError: ConfigValue 'a' is currently immutable
       >>> print(config.a)
       1

    To make them mutable again,

    .. doctest::

       >>> locker.release_lock()
       >>> config.a = 5
       >>> print(config.a)
       5

    Note that this can be used as a context manager as well:

    .. doctest::

       >>> with MarkImmutable(config.get('a'), config.get('b')):
       ...     config.a = 10
       Traceback (most recent call last):
          ...
       RuntimeError: ConfigValue 'a' is currently immutable
       >>> print(config.a)
       5
       >>> config.a = 10
       >>> print(config.a)
       10

    """

    def __init__(self, *args):
        self._targets = args
        self._locked = []
        self.lock()

    def lock(self):
        try:
            for cfg in self._targets:
                if type(cfg) is not ConfigValue:
                    if isinstance(cfg, ConfigValue):
                        # Resolve any UninitializedConfigValue
                        cfg._data
                    else:
                        raise ValueError(
                            'Only ConfigValue instances can be marked immutable.'
                        )
                cfg.__class__ = ImmutableConfigValue
                self._locked.append(cfg)
        except:
            self.release_lock()
            raise

    def release_lock(self):
        for arg in self._locked:
            arg.__class__ = ConfigValue
        self._locked = []

    def __enter__(self):
        if not self._locked:
            self.lock()
        return self

    def __exit__(self, t, v, tb):
        self.release_lock()


class ConfigList(ConfigBase, Sequence):
    """Store and manipulate a list of configuration values.

    Parameters
    ----------
    default: optional
        The default value that this ConfigList will take if no value is
        provided.  If default is a list or ConfigList, then each member
        is cast to the ConfigList's domain to build the default value,
        otherwise the default is cast to the domain and forms a default
        list with a single element.

    domain: Callable, optional
        The domain can be any callable that accepts a candidate value
        and returns the value converted to the desired type, optionally
        performing any data validation.  The result will be stored /
        added to the ConfigList.  Examples include type constructors
        like `int` or `float`.  More complex domain examples include
        callable objects; for example, the :py:class:`In` class that
        ensures that the value falls into an acceptable set or even a
        complete :py:class:`ConfigDict` instance.

    description: str, optional
        The short description of this list

    doc: str, optional
        The long documentation string for this list

    visibility: int, optional
        The visibility of this ConfigList when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigLists with visibility=0 (the
        default) will always be printed / included.  ConfigLists
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    __slots__ = ()

    def __init__(
        self, default=None, domain=None, description=None, doc=None, visibility=0
    ):
        if domain is None:
            domain = ConfigValue()
        elif isinstance(domain, ConfigBase):
            pass
        else:
            domain = ConfigValue(None, domain=domain)
        ConfigBase.__init__(
            self,
            default=default,
            domain=domain,
            description=description,
            doc=doc,
            visibility=visibility,
        )

    def __setstate__(self, state):
        state = super(ConfigList, self).__setstate__(state)
        for x in self._data:
            x._parent = self

    def __getitem__(self, key):
        val = self._data[key]
        self._userAccessed = True
        if isinstance(val, ConfigValue):
            return val.value()
        else:
            return val

    def get(self, key, default=NOTSET):
        # Note: get() is borrowed from ConfigDict for cases where we
        # want the raw stored object (and to avoid the implicit
        # conversion of ConfigValue members to their stored data).
        try:
            val = self._data[key]
            self._userAccessed = True
            return val
        except IndexError:
            if default is NOTSET:
                raise
        # Note: self._domain is ALWAYS derived from ConfigBase
        return self._domain(default)

    def __setitem__(self, key, val):
        # Note: this will fail if the element doesn't exist in _data.
        # As a result, *this* list doesn't change when someone tries to
        # change an element; instead, the *element* gets its _userSet
        # flag set.
        # self._userSet = True
        self._data[key].set_value(val)

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        self._userAccessed = True
        return iter(self[i] for i in range(len(self._data)))

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return [config.value(accessValue) for config in self._data]

    def _setter(self, value):
        # If the _setter fails part-way through the list values, we
        # want to restore a deterministic state.  That is, either
        # set_value succeeds completely, or else nothing happens.
        _data = []
        if isinstance(value, str):
            value = list(_default_string_list_lexer(value))
        if (type(value) is list) or isinstance(value, ConfigList):
            for val in value:
                self._append(_data, val)
        else:
            self._append(_data, value)
        self._data = _data

    def set_value(self, value):
        self._setter(value)
        self._userSet = True
        for _data in self._data:
            _data._userSet = True

    def _append(self, _data, value):
        val = self._cast(value)
        if val is None:
            return
        val._parent = self
        val._name = f'[{len(_data)}]'
        # We need to reset the _userSet to False because the List domain
        # is a ConfigValue and __call__ will trigger set_value(), which
        # will set the _userSet flag.  As we get here during _default
        # processing, we want to clear that flag.  If this is actually
        # getting triggered through set_value() / append(), then
        # append() will be responsible for setting _userSet.
        val._userSet = False
        _data.append(val)

    def append(self, value=NOTSET):
        self._append(self._data, value)
        self._data[-1]._userSet = True
        # Adding something to the container should not change the
        # userSet on the container (see Pyomo/pyomo#352; now
        # Pyomo/pysp#8 for justification)
        # self._userSet = True

    @deprecated("ConfigList.add() has been deprecated.  Use append()", version='5.7.2')
    def add(self, value=NOTSET):
        "Append the specified value to the list, casting as necessary."
        return self.append(value)

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        if docMode:
            # In documentation mode, we do NOT list the documentation
            # for any sub-data, and instead document the *domain*
            # information (as all the entries should share the same
            # domain, potentially duplicating that documentation is
            # somewhat redundant, and worse, if the list is empty, then
            # no documentation is generated at all!)
            yield (level, prefix, None, self)
            subDomain = self._domain._data_collector(
                level + 1, '- ', visibility, docMode
            )
            # Pop off the (empty) block entry
            next(subDomain)
            for v in subDomain:
                yield v
            return
        if prefix:
            if not self._data:
                yield (level, prefix, [], self)
            else:
                yield (level, prefix, None, self)
                if level is not None:
                    level += 1
        for value in self._data:
            for v in value._data_collector(level, '- ', visibility, docMode):
                yield v


ConfigList._UninitializedClass = type(
    'UninitializedConfigList', (UninitializedMixin, ConfigList), {'__slots__': ()}
)


class ConfigDict(ConfigBase, Mapping):
    """Store and manipulate a dictionary of configuration values.

    Parameters
    ----------
    description: str, optional
        The short description of this list

    doc: str, optional
        The long documentation string for this list

    implicit: bool, optional
        If True, the ConfigDict will allow "implicitly" declared
        keys, that is, keys can be stored into the ConfigDict that
        were not previously declared using :py:meth:`declare` or
        :py:meth:`declare_from`.

    implicit_domain: Callable, optional
        The domain that will be used for any implicitly-declared keys.
        Follows the same rules as :py:meth:`ConfigValue`'s `domain`.

    visibility: int, optional
        The visibility of this ConfigDict when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigDicts with visibility=0 (the
        default) will always be printed / included.  ConfigDicts
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    content_filters = {None, 'all', 'userdata'}

    # Note: __getstate__ relies on this field ordering.  Do not change.
    __slots__ = ('_implicit_domain', '_declared', '_implicit_declaration')
    _reserved_words = set()

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        self._declared = set()
        self._implicit_declaration = implicit
        if (
            implicit_domain is None
            or type(implicit_domain) is DynamicImplicitDomain
            or isinstance(implicit_domain, ConfigBase)
        ):
            self._implicit_domain = implicit_domain
        else:
            self._implicit_domain = ConfigValue(None, domain=implicit_domain)
        ConfigBase.__init__(
            self,
            None,
            dict if self.__class__ is ConfigDict else self.__class__,
            description,
            doc,
            visibility,
        )
        self._data = {}

    def __getstate__(self):
        state = super().__getstate__()
        state.append(_picklable(self._implicit_domain, self))
        # Note: [1:] intentionally skips the _implicit_domain (which we
        # just handled)
        state.extend(getattr(self, key) for key in ConfigDict.__slots__[1:])
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        for key, val in zip(ConfigDict.__slots__, state[len(ConfigBase.__slots__) :]):
            object.__setattr__(self, key, val)
        for x in self._data.values():
            x._parent = self

    def __dir__(self):
        # Note that dir() returns the *normalized* names (i.e., no spaces)
        return sorted(super(ConfigDict, self).__dir__() + list(self._data))

    def __getitem__(self, key):
        self._userAccessed = True
        _key = str(key).replace(' ', '_')
        if isinstance(self._data[_key], ConfigValue):
            return self._data[_key].value()
        else:
            return self._data[_key]

    def get(self, key, default=NOTSET):
        self._userAccessed = True
        _key = str(key).replace(' ', '_')
        if _key in self._data:
            return self._data[_key]
        if default is NOTSET:
            return None
        if self._implicit_domain is not None:
            if type(self._implicit_domain) is DynamicImplicitDomain:
                return self._implicit_domain(key, default)
            else:
                return self._implicit_domain(default)
        else:
            return ConfigValue(default)

    def setdefault(self, key, default=NOTSET):
        self._userAccessed = True
        _key = str(key).replace(' ', '_')
        if _key in self._data:
            return self._data[_key]
        if default is NOTSET:
            return self.add(key, None)
        else:
            return self.add(key, default)

    def __setitem__(self, key, val):
        _key = str(key).replace(' ', '_')
        if _key not in self._data:
            self.add(key, val)
        else:
            cfg = self._data[_key]
            # Trap self-assignment (useful for providing editor completion)
            if cfg is val:
                return
            cfg.set_value(val)

    def __delitem__(self, key):
        # Note that this will produce a KeyError if the key is not valid
        # for this ConfigDict.
        _key = str(key).replace(' ', '_')
        del self._data[_key]
        # Clean up the other data structures
        self._declared.discard(_key)

    def __contains__(self, key):
        _key = str(key).replace(' ', '_')
        return _key in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return map(attrgetter('_name'), self._data.values())

    def __getattr__(self, attr):
        # Note: __getattr__ is only called after all "usual" attribute
        # lookup methods have failed.  So, if we get here, we already
        # know that key is not a __slot__ or a method, etc...
        _attr = attr.replace(' ', '_')
        # Note: we test for "_data" because finding attributes on a
        # partially constructed ConfigDict (before the _data attribute
        # was declared) can lead to infinite recursion.
        if _attr == "_data" or _attr not in self._data:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )
        return ConfigDict.__getitem__(self, _attr)

    def __setattr__(self, name, value):
        if name in ConfigDict._reserved_words:
            super().__setattr__(name, value)
        else:
            ConfigDict.__setitem__(self, name, value)

    def __delattr__(self, name):
        _key = str(name).replace(' ', '_')
        if _key in self._data:
            del self[_key]
        elif _key in dir(self):
            raise AttributeError(
                "'%s' object attribute '%s' is read-only" % (type(self).__name__, name)
            )
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (type(self).__name__, name)
            )

    def keys(self):
        return iter(self)

    def values(self):
        self._userAccessed = True
        return map(self.__getitem__, self._data)

    def items(self):
        self._userAccessed = True
        for key, val in self._data.items():
            yield (val._name, self[key])

    @deprecated('The iterkeys method is deprecated. Use dict.keys().', version='6.0')
    def iterkeys(self):
        return self.keys()

    @deprecated('The itervalues method is deprecated. Use dict.keys().', version='6.0')
    def itervalues(self):
        return self.values()

    @deprecated('The iteritems method is deprecated. Use dict.keys().', version='6.0')
    def iteritems(self):
        return self.items()

    def _add(self, name, config):
        name = str(name)
        _name = name.replace(' ', '_')
        if config._parent is not None:
            raise ValueError(
                "config '%s' is already assigned to ConfigDict '%s'; "
                "cannot reassign to '%s'"
                % (name, config._parent.name(True), self.name(True))
            )
        if _name in self._data:
            raise ValueError(
                "duplicate config '%s' defined for ConfigDict '%s'"
                % (name, self.name(True))
            )
        self._data[_name] = config
        config._parent = self
        config._name = name
        return config

    def declare(self, name, config):
        _name = str(name).replace(' ', '_')
        ans = self._add(name, config)
        self._declared.add(_name)
        return ans

    def declare_from(self, other, skip=None):
        if not isinstance(other, ConfigDict):
            raise ValueError("ConfigDict.declare_from() only accepts other ConfigDicts")
        # Note that we duplicate ["other()"] other so that this
        # ConfigDict's entries are independent of the other's
        for key in other.keys():
            if skip and key in skip:
                continue
            if key in self:
                raise ValueError(
                    "ConfigDict.declare_from passed a block "
                    "with a duplicate field, '%s'" % (key,)
                )
            self.declare(key, other.get(key)())

    def add(self, name, config):
        if not self._implicit_declaration:
            raise ValueError(
                "Key '%s' not defined in ConfigDict '%s'"
                " and Dict disallows implicit entries" % (name, self.name(True))
            )

        if self._implicit_domain is None:
            if isinstance(config, ConfigBase):
                ans = self._add(name, config)
            else:
                ans = self._add(name, ConfigValue(config))
        elif type(self._implicit_domain) is DynamicImplicitDomain:
            ans = self._add(name, self._implicit_domain(name, config))
        else:
            ans = self._add(name, self._implicit_domain(config))
        ans._userSet = True
        # Adding something to the container should not change the
        # userSet on the container (see Pyomo/pyomo#352; now
        # Pyomo/pysp#8 for justification)
        # self._userSet = True
        return ans

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return {cfg._name: cfg.value(accessValue) for cfg in self._data.values()}

    def set_value(self, value, skip_implicit=False):
        if value is None:
            return self
        if isinstance(value, str):
            value = dict(_default_string_dict_lexer(value))
        if (type(value) is not dict) and (not isinstance(value, ConfigDict)):
            raise ValueError(
                "Expected dict value for %s.set_value, found %s"
                % (self.name(True), type(value).__name__)
            )
        if not value:
            return self
        _implicit = []
        _decl_map = {}
        for key in value:
            _key = str(key).replace(' ', '_')
            if _key in self._data:
                # str(key) may not be key... store the mapping so that
                # when we later iterate over the _data, we can map
                # the local keys back to the incoming value keys.
                _decl_map[_key] = key
            else:
                if skip_implicit:
                    pass
                elif self._implicit_declaration:
                    _implicit.append(key)
                else:
                    raise ValueError(
                        "key '%s' not defined for ConfigDict '%s' and "
                        "implicit (undefined) keys are not allowed"
                        % (key, self.name(True))
                    )

        # If the set_value fails part-way through the new values, we
        # want to restore a deterministic state.  That is, either
        # set_value succeeds completely, or else nothing happens.
        _old_data = self.value(False)
        try:
            # We want to set the values in declaration order (so that
            # things are deterministic and in case a validation depends
            # on the order)
            for key in self._data:
                if key in _decl_map:
                    self[key] = value[_decl_map[key]]
            # implicit data is declared at the end (in sorted order)
            for key in sorted(_implicit):
                self.add(key, value[key])
        except:
            self.reset()
            self.set_value(_old_data)
            raise
        self._userSet = True
        return self

    def reset(self):
        # Reset the values in the order they were declared.  This
        # allows reset functions to have a deterministic ordering.
        for key, val in list(self._data.items()):
            if key in self._declared:
                val.reset()
            else:
                del self._data[key]
        self._userAccessed = False
        self._userSet = False

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        if prefix:
            yield (level, prefix, None, self)
            if level is not None:
                level += 1
        for cfg in self._data.values():
            yield from cfg._data_collector(level, cfg._name + ': ', visibility, docMode)


ConfigDict._UninitializedClass = None
ConfigDict._reserved_words.update(dir(ConfigDict))

# Backwards compatibility: ConfigDict was originally named ConfigBlock.
ConfigBlock = ConfigDict
