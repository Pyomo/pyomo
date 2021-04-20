#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections.abc import Mapping
import inspect
import importlib
import logging
import sys

from pyomo.common.deprecation import deprecation_warning

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

    # We need special handling for Sphinx here, as it will look for the
    # __sphinx_mock__ attribute on all module-level objects, and we need
    # that to raise an AttributeError and not a DeferredImportError
    _getattr_raises_attributeerror = {'__sphinx_mock__',}

    def __init__(self, name, message, version_error, import_error):
        self.__name__ = name
        self._moduleunavailable_info_ = (message, version_error, import_error)

    def __getattr__(self, attr):
        if attr in ModuleUnavailable._getattr_raises_attributeerror:
            raise AttributeError("'%s' object has no attribute '%s'"
                                 % (type(self).__name__, attr))
        raise DeferredImportError(self._moduleunavailable_message())

    def _moduleunavailable_message(self, msg=None):
        _err, _ver, _imp = self._moduleunavailable_info_
        if msg is None:
            msg = _err
        if _imp:
            if not msg:
                msg = (
                    "The %s module (an optional Pyomo dependency) " \
                    "failed to import: %s" % (self.__name__, _imp)
                )
            else:
                msg += " (import raised %s)" % (_imp,)
        if _ver:
            if not msg:
                msg = "The %s module %s" % (self.__name__, _ver)
            else:
                msg += " (%s)" % (_ver,)
        return msg

    def generate_import_warning(self, logger='pyomo.common', msg=None):
        logging.getLogger(logger).warning(self._moduleunavailable_message(msg))

class DeferredImportModule(object):
    """Dummy object that serves as a module placeholder until the first time
    getattr is called, at which point it imports the module and returns
    the module attribute.

    This object is returned by attempt_import() in lieu of the module in
    the case that the module import fails.  Any attempts to access
    attributes on this object will raise a DeferredImportError
    exception.
    """
    def __init__(self, indicator, deferred_submodules, submodule_name):
        self._indicator_flag = indicator
        self._submodule_name = submodule_name

        if not deferred_submodules:
            return
        if submodule_name is None:
            submodule_name = ''
        for name in deferred_submodules:
            if not name.startswith(submodule_name + '.'):
                continue
            _local_name = name[(1+len(submodule_name)):]
            if '.' in _local_name:
                continue
            setattr(self, _local_name, DeferredImportModule(
                indicator, deferred_submodules,
                submodule_name + '.' + _local_name))

    def __getattr__(self, attr):
        self._indicator_flag.resolve()
        _mod = self._indicator_flag._module
        if self._submodule_name:
            for _sub in self._submodule_name[1:].split('.'):
                _mod = getattr(_mod, _sub)
        return getattr(_mod, attr)


class _DeferredImportIndicatorBase(object):
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

    def __init__(self, name, error_message, catch_exceptions,
                 minimum_version, original_globals, callback, importer,
                 deferred_submodules):
        self._names = [name]
        for _n in tuple(self._names):
            if '.' in _n:
                self._names.append(_n.split('.')[-1])
        self._error_message = error_message
        self._catch_exceptions = catch_exceptions
        self._minimum_version = minimum_version
        self._original_globals = original_globals
        self._callback = callback
        self._importer = importer
        self._module = None
        self._available = None
        self._deferred_submodules = deferred_submodules

    def __bool__(self):
        self.resolve()
        return self._available

    def resolve(self):
        if self._module is None:
            try:
                # Only attempt the import once
                self._module, self._available = attempt_import(
                    name=self._names[0],
                    error_message=self._error_message,
                    catch_exceptions=self._catch_exceptions,
                    minimum_version=self._minimum_version,
                    callback=self._callback,
                    importer=self._importer,
                    defer_check=False,
                )
            except Exception as e:
                # make sure that we cache the result
                self._module = ModuleUnavailable(
                    self._names[0],
                    "Exception raised when importing %s" % (self._names[0],),
                    None,
                    "%s: %s" % (type(e).__name__, e),
                )
                self._available = False
                raise

            # If this module was not found, then we need to check for
            # deferred submodules and resolve them as well
            if self._deferred_submodules and \
               type(self._module) is ModuleUnavailable:
                info = self._module._moduleunavailable_info_
                for submod in self._deferred_submodules:
                    refmod = self._module
                    for name in submod.split('.')[1:]:
                        try:
                            refmod = getattr(refmod, name)
                        except DeferredImportError:
                            setattr(refmod, name, ModuleUnavailable(
                                refmod.__name__+submod, *info))
                            refmod = getattr(refmod, name)

            # Replace myself in the original globals() where I was
            # declared
            self.replace_self_in_globals(self._original_globals)

        # Replace myself in the caller globals (to avoid calls to
        # this method in the future)
        _frame = inspect.currentframe().f_back
        while _frame.f_globals is globals():
            _frame = _frame.f_back
        self.replace_self_in_globals(_frame.f_globals)

    def replace_self_in_globals(self, _globals):
        for k,v in _globals.items():
            if v is self:
                _globals[k] = self._available
            elif v.__class__ is DeferredImportModule and \
                 v._indicator_flag is self:
                if v._submodule_name is None:
                    _globals[k] = self._module
                else:
                    _mod_path = v._submodule_name.split('.')[1:]
                    _mod = self._module
                    for _sub in _mod_path:
                        _mod = getattr(_mod, _sub)
                    _globals[k] = _mod


class _DeferredAnd(_DeferredImportIndicatorBase):
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __bool__(self):
        return bool(self._a) and bool(self._b)


class _DeferredOr(_DeferredImportIndicatorBase):
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __bool__(self):
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


def attempt_import(name, error_message=None, only_catch_importerror=None,
                   minimum_version=None, alt_names=None, callback=None,
                   importer=None, defer_check=True, deferred_submodules=None,
                   catch_exceptions=None):

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
        DEPRECATED: alt_names no longer needs to be specified and is ignored.
        A list of common alternate names by which to look for this
        module in the globals() namespaces.  For example, the alt_names
        for NumPy would be ['np'].  (deprecated in version 6.0)

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

    deferred_submodules: Iterable, optional
        If provided, an iterable of submodule names within this module that
        can be accessed without triggering a deferred import of this
        module.  For example, the deferred_submodules for matplotlib is
        ``['pyplot']``

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
    if alt_names is not None:
        deprecation_warning('alt_names no longer needs to be specified '
                            'and is ignored', version='TBD')

    if only_catch_importerror is not None:
        deprecation_warning(
            "only_catch_importerror is deprecated.  Pass exceptions to "
            "catch using the catch_exceptions argument", version='5.7.3')
        if catch_exceptions is not None:
            raise ValueError("Cannot specify both only_catch_importerror "
                             "and catch_exceptions")
        if only_catch_importerror:
            catch_exceptions = (ImportError,)
        else:
            catch_exceptions = (ImportError, Exception)
    if catch_exceptions is None:
        catch_exceptions = (ImportError,)

    # If we are going to defer the check until later, return the
    # deferred import module object
    if defer_check:
        if deferred_submodules:
            if isinstance(deferred_submodules, Mapping):
                deprecation_warning(
                    'attempt_import(): deferred_submodules takes an iterable '
                    'and not a mapping (the alt_names supplied by the mapping '
                    'are no longer needed and are ignored).', version='TBD')
                deferred_submodules = list(deferred_submodules)

            # Ensures all names begin with '.'
            #
            # Fill in any missing submodules.  For example, if a user
            # provides {'foo.bar.baz': ['bz']}, then expand the dict to
            # {'.foo': None, '.foo.bar': None, '.foo.bar.baz': ['bz']}
            deferred = []
            for _submod in deferred_submodules:
                if _submod[0] != '.':
                    _submod = '.' + _submod
                _mod_path = _submod.split('.')
                for i in range(len(_mod_path)):
                    _test_mod = '.'.join(_mod_path[:i])
                    if _test_mod not in deferred:
                        deferred.append(_test_mod)
                deferred.append(_submod)
            deferred = [_ for _ in deferred if _]
        else:
            deferred = None

        indicator = DeferredImportIndicator(
            name=name,
            error_message=error_message,
            catch_exceptions=catch_exceptions,
            minimum_version=minimum_version,
            original_globals=inspect.currentframe().f_back.f_globals,
            callback=callback,
            importer=importer,
            deferred_submodules=deferred)
        return DeferredImportModule(indicator, deferred, None), indicator

    if deferred_submodules:
        raise ValueError(
            "deferred_submodules is only valid if defer_check==True")

    import_error = None
    version_error = None
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
        else:
            version = getattr(module, '__version__', 'UNKNOWN')
            version_error = (
                "version %s does not satisfy the minimum version %s"
                % (version, minimum_version))
    except catch_exceptions as e:
        import_error = "%s: %s" % (type(e).__name__, e)

    module = ModuleUnavailable(name, error_message, version_error, import_error)
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
        import scipy.stats
        # As of scipy 1.6.0, importing scipy.stats causes the following
        # to be automatically imported.  However, we will still
        # explicitly import them here to guard against potential future
        # changes in scipy.
        import scipy.integrate
        import scipy.sparse
        import scipy.spatial

def _finalize_pympler(module, available):
    if available:
        # Import key subpackages that we will want to assume are present
        import pympler.muppy

def _finalize_matplotlib(module, available):
    if not available:
        return
    # You must switch matplotlib backends *before* importing pyplot.  If
    # we are in the middle of testing, we need to switch the backend to
    # 'Agg', otherwise attempts to generate plots on CI services without
    # terminal windows will fail.
    if any(mod in sys.modules for mod in ('nose', 'nose2', 'sphinx')):
        module.use('Agg')
    import matplotlib.pyplot

yaml, yaml_available = attempt_import(
    'yaml', callback=_finalize_yaml)
pympler, pympler_available = attempt_import(
    'pympler', callback=_finalize_pympler)
numpy, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import(
    'scipy', callback=_finalize_scipy,
    deferred_submodules=['stats', 'sparse', 'spatial', 'integrate'])
networkx, networkx_available = attempt_import('networkx')
pandas, pandas_available = attempt_import('pandas')
dill, dill_available = attempt_import('dill')

# Note that matplotlib.pyplot can generate a runtime error on OSX when
# not installed as a Framework (as is the case in the CI systems)
matplotlib, matplotlib_available = attempt_import(
    'matplotlib',
    callback=_finalize_matplotlib,
    deferred_submodules=['pyplot', 'pylab'],
    catch_exceptions=(ImportError, RuntimeError),
)

try:
    import cPickle as pickle
except ImportError:
    import pickle
