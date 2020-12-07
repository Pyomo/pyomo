#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import inspect
import importlib
import logging

class DeferredImportError(ImportError):
    pass

class ModuleUnavailable(object):
    """Dummy object that raises a DeferredImportError upon attribute access

    This object is returned by attempt_import() in lieu of the module in
    the case that the module import fails.  Any attempts to access
    attributes on this object will raise a DeferredImportError
    exception.

    Parameters
    ----------
    message: str
        The string message to return in the raised exception
    """
    def __init__(self, message):
        self._error_message_ = message

    def __getattr__(self, attr):
        raise DeferredImportError(self._error_message_)

    def generate_import_warning(self, logger='pyomo.common'):
        logging.getLogger(logger).warning(
            self._error_message_)

class DeferredImportModule(object):
    """Dummy object that serves as a module placeholder until the first time
    getattr is called, at which point it imports the module and returns
    the module attribute.

    This object is returned by attempt_import() in lieu of the module in
    the case that the module import fails.  Any attempts to access
    attributes on this object will raise a DeferredImportError
    exception.
    """
    def __init__(self, indicator):
        self._indicator_flag = indicator

    def __getattr__(self, attr):
        self._indicator_flag.resolve()
        return getattr(self._indicator_flag._module, attr)


class _DeferredImportIndicatorBase(object):
    def __bool__(self):
        return self.__nonzero__()

    def __and__(self, other):
        return _DeferredAnd(self, other)

    def __or__(self, other):
        return _DeferredOr(self, other)

    def __rand__(self, other):
        return _DeferredAnd(other, self)

    def __ror__(self, other):
        return _DeferredOr(other, self)


class DeferredImportIndicator(_DeferredImportIndicatorBase):
    """Placeholder indicating if an import was successful.

    This object serves as a placeholder for the Boolean indicator if a
    deferred module import was successful.  Casting this instance to
    bool will cause the import to be attempted.  The actual import logic
    is here and not in the DeferredImportModule to reduce the number of
    attributes on the DeferredImportModule.
    """

    def __init__(self, name, alt_names, error_message, only_catch_importerror,
                 minimum_version, original_globals, callback, importer):
        self._names = [name]
        if alt_names:
            self._names += list(alt_names)
        for _n in tuple(self._names):
            if '.' in _n:
                self._names.append(_n.split('.')[-1])
        self._error_message = error_message
        self._only_catch_importerror = only_catch_importerror
        self._minimum_version = minimum_version
        self._original_globals = original_globals
        self._callback = callback
        self._importer = importer
        self._module = None
        self._available = None

    def resolve(self):
        if self._module is None:
            try:
                # Only attempt the import once
                self._module, self._available = attempt_import(
                    name=self._names[0],
                    error_message=self._error_message,
                    only_catch_importerror=self._only_catch_importerror,
                    minimum_version=self._minimum_version,
                    callback=self._callback,
                    importer=self._importer,
                    defer_check=False,
                )
            except:
                # make sure that we cache the result
                self._module = ModuleUnavailable(
                    "Exception raised when importing %s" % (self._names[0],))
                self._available = False
                raise

            # Replace myself in the original globals() where I was
            # declared
            self.replace_self_in_globals(self._original_globals)

        # Replace myself in the caller globals (to avoid calls to
        # this method in the future)
        _globals = inspect.currentframe().f_back.f_back.f_globals
        self.replace_self_in_globals(_globals)

    def replace_self_in_globals(self, _globals):
        for name in self._names:
            if ( name in _globals
                 and isinstance(_globals[name], DeferredImportModule)
                 and _globals[name]._indicator_flag is self ):
                _globals[name] = self._module
            for flag_name in (name+'_available', 'has_'+name, 'have_'+name):
                if flag_name in _globals and _globals[flag_name] is self:
                    _globals[flag_name] = self._available

    def __nonzero__(self):
        self.resolve()
        return self._available


class _DeferredAnd(_DeferredImportIndicatorBase):
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __nonzero__(self):
        return bool(self._a) and bool(self._b)


class _DeferredOr(_DeferredImportIndicatorBase):
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __nonzero__(self):
        return bool(self._a) or bool(self._b)


def check_min_version(module, min_version):
    if isinstance(module, DeferredImportModule):
        indicator = module._indicator_flag
        indicator.resolve()
        if indicator._available:
            module = indicator._module
        else:
            return False
    try:
        from packaging import version as _version
        _parser = _version.parse
    except ImportError:
        # pkg_resources is an order of magnitude slower to import than
        # packaging.  Only use it if the preferred (but optional)
        # packaging library is not present
        from pkg_resources import parse_version as _parser

    version = getattr(module, '__version__', '0.0.0')
    return _parser(min_version) <= _parser(version)


def attempt_import(name, error_message=None, only_catch_importerror=True,
                   minimum_version=None, alt_names=None, callback=None,
                   importer=None, defer_check=True):

    """Attempt to import the specified module.

    This will attempt to import the specified module, returning a
    (module, available) tuple.  If the import was successful, `module`
    will be the imported module and `available` will be True.  If the
    import results in an exception, then `module` will be an instance of
    :py:class:`ModuleUnavailable` and `available` will be False

    The following is equivalent to ``import numpy as np``:

    .. doctest::

       >>> from pyomo.common.dependencies import attempt_import
       >>> numpy, numpy_available = attempt_import('numpy')

    The import can be "deferred" until the first time the code either
    attempts to access the module or checks the boolean value of the
    available flag.  This allows optional dependencies to be declared at
    the module scope but not imported until they are actually used by
    the module (thereby speeding up the initial package import).
    Deferred imports are handled by two helper classes
    (DeferredImportModule and DeferredImportIndicator).  Upon actual
    import, DeferredImportIndicator.resolve() attempts to replace those
    objects (in both the local and original global namespaces) with the
    imported module and boolean flag so that subsequent uses of the
    module do not incur any overhead due to the delayed import.

    Parameters
    ----------
    name: str
        The name of the module to import

    error_message: str, optional
        The message for the exception raised by ModuleUnavailable

    only_catch_importerror: bool, optional
        If True (the default), exceptions other than ImportError raised
        during module import will be reraised.  If False, any exception
        will result in returning a ModuleUnavailable object.

    minimum_version: str, optional
        The minimum acceptable module version (retrieved from
        module.__version__)

    alt_names: list, optional
        A list of common alternate names by which to look for this
        module in the globals() namespaces.  For example, the alt_names
        for NumPy would be ['np']

    callback: function, optional
        A function with the signature "`fcn(module, available)`" that
        will be called after the import is first attempted.

    importer: function, optional
        A function that will perform the import and return the imported
        module (or raise an ImportError).  This is useful for cases
        where there are several equivalent modules and you want to
        import/return the first one that is available.

    defer_check: bool, optional
        If True (the default), then the attempted import is deferred
        until the first use of either the module or the availability
        flag.  The method will return instances of DeferredImportModule
        and DeferredImportIndicator.

    Returns
    -------
    : module
        the imported module, or an instance of
        :py:class:`ModuleUnavailable`, or an instance of
        :py:class:`DeferredImportModule`
    : bool
        Boolean indicating if the module import succeeded or an instance
        of "py:class:`DeferredImportIndicator`

    """
    # If we are going to defer the check until later, return the
    # deferred import module object
    if defer_check:
        indicator = DeferredImportIndicator(
            name=name,
            alt_names=alt_names,
            error_message=error_message,
            only_catch_importerror=only_catch_importerror,
            minimum_version=minimum_version,
            original_globals=inspect.currentframe().f_back.f_globals,
            callback=callback,
            importer=importer)
        return DeferredImportModule(indicator), indicator

    try:
        if importer is None:
            module = importlib.import_module(name)
        else:
            module = importer()
        if ( minimum_version is None
             or check_min_version(module, minimum_version) ):
            if callback is not None:
                callback(module, True)
            return module, True
        elif error_message:
            version = getattr(module, '__version__', 'UNKNOWN')
            error_message += " (version %s does not satisfy the minimum " \
                             "version %s)" % (version, minimum_version)
        else:
            version = getattr(module, '__version__', 'UNKNOWN')
            error_message = "The %s module version %s does not satisfy " \
                            "the minimum version %s" % (
                                name, version, minimum_version)
    except ImportError:
        pass
    except:
        if only_catch_importerror:
            raise

    if not error_message:
        error_message = "The %s module (an optional Pyomo dependency) " \
                        "failed to import" % (name,)

    module = ModuleUnavailable(error_message)
    if callback is not None:
        callback(module, False)
    return module, False

#
# Common optional dependencies used throughout Pyomo
#

yaml_load_args = {}
def _finalize_yaml(module, available):
    # Recent versions of PyYAML issue warnings if the Loader argument is
    # not set
    if available and hasattr(module, 'SafeLoader'):
        yaml_load_args['Loader'] = module.SafeLoader

def _finalize_scipy(module, available):
    if available:
        # Import key subpackages that we will want to assume are present
        import scipy.sparse
        import scipy.spatial
        import scipy.stats

def _finalize_pympler(module, available):
    if available:
        # Import key subpackages that we will want to assume are present
        import pympler.muppy

yaml, yaml_available = attempt_import('yaml', callback=_finalize_yaml)
pympler, pympler_available = attempt_import(
    'pympler', callback=_finalize_pympler)
numpy, numpy_available = attempt_import('numpy', alt_names=['np'])
scipy, scipy_available = attempt_import('scipy', callback=_finalize_scipy)
networkx, networkx_available = attempt_import('networkx', alt_names=['nx'])
pandas, pandas_available = attempt_import('pandas', alt_names=['pd'])
dill, dill_available = attempt_import('dill')
