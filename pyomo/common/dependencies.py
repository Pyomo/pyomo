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

import inspect
import importlib
import importlib.util
import logging
import sys
import warnings

from collections.abc import Mapping
from types import ModuleType
from typing import List

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeferredImportError
from pyomo.common.flags import (
    in_testing_environment,
    building_documentation,
    serializing,
)

SUPPRESS_DEPENDENCY_WARNINGS = False


class ModuleUnavailable(object):
    """Mock object that raises :py:class:`.DeferredImportError` upon attribute access

    This object is returned by :py:func:`attempt_import()` in lieu of
    the module in the case that the module import fails.  Any attempts
    to access attributes on this object will raise a :py:class:`.DeferredImportError`
    exception.

    Parameters
    ----------
    name: str
        The module name that was being imported

    message: str
        The string message to return in the raised exception

    version_error: str
        A string to add to the message if the module failed to import because
        it did not match the required version

    import_error: str
        A string to add to the message documenting the Exception
        raised when the module failed to import.

    package: str
        The module name that originally attempted the import
    """

    _getattr_raises_attributeerror = {
        # We need special handling for Sphinx here, as it will look for the
        # __sphinx_mock__ attribute on all module-level objects, and we need
        # that to raise an AttributeError and not a DeferredImportError
        '__sphinx_mock__',
        # We need special handling for dill as well, as dill attempts to
        # pickle module globals by looking for the '_dill' attribute on
        # all global objects.
        '_dill',
    }

    def __init__(self, name, message, version_error, import_error, package):
        self.__name__ = name
        self._moduleunavailable_info_ = (message, version_error, import_error, package)

    def __getattr__(self, attr):
        if serializing() or attr in ModuleUnavailable._getattr_raises_attributeerror:
            msg = "'%s' object has no attribute '%s'" % (type(self).__name__, attr)
            raise AttributeError(msg)
        raise DeferredImportError(self._moduleunavailable_message())

    def __getstate__(self):
        return (self.__name__, self._moduleunavailable_info_)

    def __setstate__(self, state):
        self.__name__, self._moduleunavailable_info_ = state

    # Included because recent dill picklers look for the mro() when
    # detecting numpy types
    def mro(self):
        return [ModuleUnavailable, object]

    def _moduleunavailable_message(self, msg=None):
        _err, _ver, _imp, _package = self._moduleunavailable_info_
        if msg is None:
            msg = _err
        if _imp:
            if not msg or not str(msg):
                _pkg_str = _package.split('.')[0].capitalize()
                if _pkg_str:
                    _pkg_str += ' '
                msg = (
                    "The %s module (an optional %sdependency) "
                    "failed to import: %s" % (self.__name__, _pkg_str, _imp)
                )
            else:
                msg = "%s (import raised %s)" % (msg, _imp)
        if _ver:
            if not msg or not str(msg):
                msg = "The %s module %s" % (self.__name__, _ver)
            else:
                msg = "%s (%s)" % (msg, _ver)
        return msg

    def log_import_warning(self, logger='pyomo', msg=None):
        """Log the import error message to the specified logger

        This will log the the import error message to the specified
        logger.  If ``msg=`` is specified, it will override the default
        message passed to this instance of
        :py:class:`ModuleUnavailable`.

        """
        logging.getLogger(logger).warning(self._moduleunavailable_message(msg))

    @deprecated("use :py:class:`log_import_warning()`", version='6.0')
    def generate_import_warning(self, logger='pyomo.common'):
        self.log_import_warning(logger)


class DeferredImportModule(object):
    """Mock module object to support the deferred import of a module.

    This object is returned by :py:func:`attempt_import()` in lieu of
    the module when :py:func:`attempt_import()` is called with
    ``defer_import=True``.  Any attempts to access attributes on this
    object will trigger the actual module import and return either the
    appropriate module attribute or else if the module import fails,
    raise a :py:class:`.DeferredImportError` exception.

    """

    def __init__(self, indicator, deferred_submodules, submodule_name):
        self._indicator_flag = indicator
        self._submodule_name = submodule_name
        self.__file__ = None  # Disable coverage of this module
        self.__spec__ = None  # Indicate that this is not a "real" module

        if not deferred_submodules:
            return
        if submodule_name is None:
            submodule_name = ''
        for name in deferred_submodules:
            if not name.startswith(submodule_name + '.'):
                continue
            _local_name = name[(1 + len(submodule_name)) :]
            if '.' in _local_name:
                continue
            setattr(
                self,
                _local_name,
                DeferredImportModule(
                    indicator, deferred_submodules, submodule_name + '.' + _local_name
                ),
            )

    def __getattr__(self, attr):
        self._indicator_flag.resolve()
        _mod = self._indicator_flag._module
        if self._submodule_name:
            for _sub in self._submodule_name[1:].split('.'):
                _mod = getattr(_mod, _sub)
        return getattr(_mod, attr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for k, v in state.items():
            super().__setattr__(k, v)

    # Included because recent dill picklers look for the mro() when
    # detecting numpy types
    def mro(self):
        return [DeferredImportModule, object]


def UnavailableClass(unavailable_module):
    """Function to generate an "unavailable" base class

    This function returns a custom class that wraps the
    :py:class:`ModuleUnavailable` instance returned by
    :py:func:`attempt_import` when the target module is not available.
    Any attempt to instantiate this class (or a class derived from it)
    or access a class attribute will raise the
    :py:class:`.DeferredImportError` from the wrapped
    :py:class:`ModuleUnavailable` object.

    Parameters
    ----------
    unavailable_module: ModuleUnavailable
        The :py:class:`ModuleUnavailable` instance (from
        :py:func:`attempt_import`) to use to generate the
        :py:class:`.DeferredImportError`.

    Example
    -------

    Declaring a class that inherits from an optional dependency:

    .. doctest::

       >>> from pyomo.common.dependencies import attempt_import, UnavailableClass
       >>> bogus, bogus_available = attempt_import('bogus_unavailable_class')
       >>> class MyPlugin(bogus.plugin if bogus_available else UnavailableClass(bogus)):
       ...     pass

    Attempting to instantiate the derived class generates an exception
    when the module is unavailable:

    .. doctest::

       >>> MyPlugin()
       Traceback (most recent call last):
          ...
       pyomo.common.dependencies.DeferredImportError: The class 'MyPlugin' cannot be
       created because a needed optional dependency was not found (import raised
       ModuleNotFoundError: No module named 'bogus_unavailable_class')

    As does attempting to access class attributes on the derived class:

    .. testcode::
       :hide:

       # We suppress this exception when building the documentation
       # from pyomo.common.flags import building_documentation
       building_documentation(False)

    .. doctest::

       >>> MyPlugin.create_instance()
       Traceback (most recent call last):
          ...
       pyomo.common.dependencies.DeferredImportError: The class attribute
       'MyPlugin.create_instance' is not available because a needed optional
       dependency was not found (import raised ModuleNotFoundError: No module
       named 'bogus_unavailable_class')

    .. testcode::
       :hide:

       building_documentation(None)

    """

    class UnavailableMeta(type):
        def __getattr__(cls, name):
            if building_documentation():
                # If we are building documentation, avoid the
                # DeferredImportError (we will still raise one if
                # someone attempts to *create* an instance of this
                # class)
                return getattr(super(), name)
            raise DeferredImportError(
                unavailable_module._moduleunavailable_message(
                    f"The class attribute '{cls.__name__}.{name}' is not available "
                    "because a needed optional dependency was not found"
                )
            )

    class UnavailableBase(metaclass=UnavailableMeta):
        def __new__(cls, *args, **kwargs):
            raise DeferredImportError(
                unavailable_module._moduleunavailable_message(
                    f"The class '{cls.__name__}' cannot be created because a "
                    "needed optional dependency was not found"
                )
            )

    return UnavailableBase


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
    `bool` will cause the import to be attempted.  The actual import logic
    is here and not in the :py:class:`DeferredImportModule` to reduce the number of
    attributes on the :py:class:`DeferredImportModule`.

    :py:class:`DeferredImportIndicator` supports limited logical expressions
    using the ``&`` (and) and ``|`` (or) binary operators.  Creating
    these expressions does not trigger the import of the corresponding
    :py:class:`DeferredImportModule` instances, although casting the
    resulting expression to ``bool()`` will trigger any relevant
    imports.

    """

    def __init__(
        self,
        name,
        error_message,
        catch_exceptions,
        minimum_version,
        original_globals,
        callback,
        importer,
        deferred_submodules,
    ):
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
        # If this import has a callback, then record this deferred
        # import so that any direct imports of this module also trigger
        # the resolution of this DeferredImportIndicator (and the
        # corresponding callback)
        if callback is not None:
            DeferredImportCallbackFinder._callbacks.setdefault(name, []).append(self)

    def __bool__(self):
        self.resolve()
        return self._available

    def resolve(self):
        # Only attempt the import once, then cache some form of result
        if self._module is None:
            package = self._original_globals.get('__name__', '')
            try:
                self._module, self._available = _perform_import(
                    name=self._names[0],
                    error_message=self._error_message,
                    minimum_version=self._minimum_version,
                    callback=self._callback,
                    importer=self._importer,
                    catch_exceptions=self._catch_exceptions,
                    package=package,
                )
            except Exception as e:
                # make sure that we cache the result
                self._module = ModuleUnavailable(
                    self._names[0],
                    "Exception raised when importing %s" % (self._names[0],),
                    None,
                    "%s: %s" % (type(e).__name__, e),
                    package,
                )
                self._available = False
                raise

            # If this module was not found, then we need to check for
            # deferred submodules and resolve them as well
            if self._deferred_submodules and type(self._module) is ModuleUnavailable:
                info = self._module._moduleunavailable_info_
                for submod in self._deferred_submodules:
                    refmod = self._module
                    for name in submod.split('.')[1:]:
                        try:
                            refmod = getattr(refmod, name)
                        except DeferredImportError:
                            setattr(
                                refmod,
                                name,
                                ModuleUnavailable(refmod.__name__ + submod, *info),
                            )
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
        for k, v in _globals.items():
            if v is self:
                _globals[k] = self._available
            elif v.__class__ is DeferredImportModule and v._indicator_flag is self:
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

    version = getattr(module, '__version__', '0.0.0')
    return packaging.version.parse(min_version) <= packaging.version.parse(version)


#
# Note that we are duck-typing the Loader and MetaPathFinder base
# classes from importlib.abc.  This avoids a (surprisingly costly)
# import of importlib.abc
#
class DeferredImportCallbackLoader:
    """Custom Loader to resolve registered :py:class:`DeferredImportIndicator` objects

    This :py:class:`importlib.abc.Loader` loader wraps a regular loader
    and automatically resolves the registered
    :py:class:`DeferredImportIndicator` objects after the module is
    loaded.

    """

    def __init__(self, loader, deferred_indicators: List[DeferredImportIndicator]):
        self._loader = loader
        self._deferred_indicators = deferred_indicators

    def module_repr(self, module: ModuleType) -> str:
        return self._loader.module_repr(module)

    def create_module(self, spec) -> ModuleType:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        # Now that the module has been loaded, trigger the resolution of
        # the deferred indicators (and their associated callbacks)
        for deferred in self._deferred_indicators:
            deferred.resolve()

    def load_module(self, fullname) -> ModuleType:
        return self._loader.load_module(fullname)


class DeferredImportCallbackFinder:
    """Custom Finder that will wrap the normal loader to trigger callbacks

    This :py:class:`importlib.abc.MetaPathFinder` finder will wrap the
    normal loader returned by ``PathFinder`` with a loader that will
    trigger custom callbacks after the module is loaded.  We use this to
    trigger the post import callbacks registered through
    :py:func:`attempt_import` even when a user imports the target library
    directly (and not through attribute access on the
    :py:class:`DeferredImportModule`.

    """

    _callbacks = {}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._callbacks:
            return None

        spec = None
        # Continue looking for the finder that would have originally
        # loaded the deferred import module by starting at the next
        # finder in sys.meta_path (this way, we are agnostic to where
        # the module is coming from: file system, registry, etc.)
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            if hasattr(finder, 'find_spec'):
                # Support standard importlib MetaPathFinders
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                # Support for imp finders/loaders (deprecated, but
                # supported through Python 3.11)
                loader = finder.find_module(fullname, path)
                if loader is not None:
                    spec = importlib.util.spec_from_loader(fullname, loader)
                    break
        else:
            # Module not found.  Returning None will proceed to the next
            # finder (which will eventually raise a ModuleNotFoundError)
            return None
        # Override the loader to trigger the finalization callback
        # after the original loader is finished
        spec.loader = DeferredImportCallbackLoader(
            spec.loader, self._callbacks[fullname]
        )
        return spec

    def invalidate_caches(self):
        pass


_DeferredImportCallbackFinder = DeferredImportCallbackFinder()
# Insert the DeferredImportCallbackFinder at the beginning of the
# sys.meta_path so that it is found before the standard finders (so that
# we can correctly inject the resolution of the DeferredImportIndicators
# -- which triggers the needed callbacks)
sys.meta_path.insert(0, _DeferredImportCallbackFinder)


def attempt_import(
    name,
    error_message=None,
    only_catch_importerror=None,
    minimum_version=None,
    alt_names=None,
    callback=None,
    importer=None,
    defer_check=None,
    defer_import=None,
    deferred_submodules=None,
    catch_exceptions=None,
):
    """Attempt to import the specified module.

    This will attempt to import the specified module, returning a
    ``(module, available)`` tuple.  If the import was successful, ``module``
    will be the imported module and ``available`` will be True.  If the
    import results in an exception, then ``module`` will be an instance of
    :py:class:`ModuleUnavailable` and ``available`` will be False

    The following

    .. doctest::

       >>> from pyomo.common.dependencies import attempt_import
       >>> numpy, numpy_available = attempt_import('numpy')

    Is roughly equivalent to

    .. doctest::

       >>> from pyomo.common.dependencies import ModuleUnavailable
       >>> try:
       ...     import numpy
       ...     numpy_available = True
       ... except ImportError as e:
       ...     numpy = ModuleUnavailable('numpy', 'Numpy is not available',
       ...                               '', str(e), globals()['__name__'])
       ...     numpy_available = False

    The import can be "deferred" until the first time the code either
    attempts to access the module or checks the Boolean value of the
    available flag.  This allows optional dependencies to be declared at
    the module scope but not imported until they are actually used by
    the module (thereby speeding up the initial package import).
    Deferred imports are handled by two helper classes
    (:py:class:`DeferredImportModule` and
    :py:class:`DeferredImportIndicator`).  Upon actual import,
    :py:meth:`DeferredImportIndicator.resolve()` attempts to replace
    those objects (in both the local and original global namespaces)
    with the imported module and Boolean flag so that subsequent uses of
    the module do not incur any overhead due to the delayed import.

    Parameters
    ----------
    name: str
        The name of the module to import

    error_message: str, optional
        The message for the exception raised by :py:class:`ModuleUnavailable`

    only_catch_importerror: bool, optional
        DEPRECATED: use ``catch_exceptions`` instead of ``only_catch_importerror``.

        If True (the default), exceptions other than ``ImportError`` raised
        during module import will be reraised.  If False, any exception
        will result in returning a :py:class:`ModuleUnavailable` object.
        (deprecated in version 5.7.3)

    minimum_version: str, optional
        The minimum acceptable module version (retrieved from
        ``module.__version__``)

    alt_names: list, optional
        DEPRECATED: ``alt_names`` no longer needs to be specified and is ignored.

        A list of common alternate names by which to look for this
        module in the ``globals()`` namespaces.  For example, the alt_names
        for NumPy would be ``['np']``.  (deprecated in version 6.0)

    callback: Callable[[ModuleType, bool], None], optional
        A function with the signature ``fcn(module, available)`` that
        will be called after the import is first attempted.

    importer: function, optional
        A function that will perform the import and return the imported
        module (or raise an :py:class:`ImportError`).  This is useful
        for cases where there are several equivalent modules and you
        want to import/return the first one that is available.

    defer_check: bool, optional
        DEPRECATED: renamed to ``defer_import`` (deprecated in version 6.7.2)

    defer_import: bool, optional
        If True, then the attempted import is deferred until the first
        use of either the module or the availability flag.  The method
        will return instances of :py:class:`DeferredImportModule` and
        :py:class:`DeferredImportIndicator`.  If False, the import will
        be attempted immediately.  If not set, then the import will be
        deferred unless the ``name`` is already present in
        ``sys.modules``.

    deferred_submodules: Iterable[str], optional
        If provided, an iterable of submodule names within this module
        that can be accessed without triggering a deferred import of
        this module.  For example, this module uses
        ``deferred_submodules=['pyplot', 'pylab']`` for ``matplotlib``.

    catch_exceptions: Iterable[Exception], optional
        If provided, this is the list of exceptions that will be caught
        when importing the target module, resulting in
        ``attempt_import`` returning a :py:class:`ModuleUnavailable`
        instance.  The default is to only catch :py:class:`ImportError`.
        This is useful when a module can regularly return additional
        exceptions during import.

    Returns
    -------
    : module
        the imported module, or an instance of
        :py:class:`ModuleUnavailable`, or an instance of
        :py:class:`DeferredImportModule`
    : bool
        Boolean indicating if the module import succeeded or an instance
        of :py:class:`DeferredImportIndicator`

    """
    if alt_names is not None:
        deprecation_warning(
            'alt_names=%s no longer needs to be specified '
            'and is ignored' % (alt_names,),
            version='6.0',
        )

    if only_catch_importerror is not None:
        deprecation_warning(
            "only_catch_importerror is deprecated.  Pass exceptions to "
            "catch using the catch_exceptions argument",
            version='5.7.3',
        )
        if catch_exceptions is not None:
            raise ValueError(
                "Cannot specify both only_catch_importerror and catch_exceptions"
            )
        if only_catch_importerror:
            catch_exceptions = (ImportError,)
        else:
            catch_exceptions = (ImportError, Exception)
    if catch_exceptions is None:
        catch_exceptions = (ImportError,)

    if defer_check is not None:
        deprecation_warning(
            'defer_check=%s is deprecated.  Please use defer_import' % (defer_check,),
            version='6.7.2',
        )
        assert defer_import is None
        defer_import = defer_check

    # If the module has already been imported, there is no reason to
    # further defer things: just import it.
    if defer_import is None:
        if name in sys.modules:
            defer_import = False
            deferred_submodules = None
        else:
            defer_import = True

    # If we are going to defer the check until later, return the
    # deferred import module object
    if defer_import:
        if deferred_submodules:
            if isinstance(deferred_submodules, Mapping):
                deprecation_warning(
                    'attempt_import(): deferred_submodules takes an iterable '
                    'and not a mapping (the alt_names supplied by the mapping '
                    'are no longer needed and are ignored).',
                    version='6.0',
                )
                deferred_submodules = list(deferred_submodules)

            # Ensures all names begin with '.'
            #
            # Fill in any missing submodules.  For example, if a user
            # provides ['foo.bar.baz'], then expand the list to
            # ['.foo', '.foo.bar', '.foo.bar.baz']
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
            deferred_submodules=deferred,
        )
        return DeferredImportModule(indicator, deferred, None), indicator

    if deferred_submodules:
        raise ValueError("deferred_submodules is only valid if defer_import==True")

    return _perform_import(
        name=name,
        error_message=error_message,
        minimum_version=minimum_version,
        callback=callback,
        importer=importer,
        catch_exceptions=catch_exceptions,
        package=inspect.currentframe().f_back.f_globals.get('__name__', ''),
    )


def _perform_import(
    name, error_message, minimum_version, callback, importer, catch_exceptions, package
):
    import_error = None
    version_error = None
    try:
        with warnings.catch_warnings():
            # Temporarily suppress all warnings: we assume we are
            # importing a third-party package here and we don't want to
            # see them?
            if SUPPRESS_DEPENDENCY_WARNINGS and not name.startswith('pyomo.'):
                warnings.resetwarnings()
                warnings.simplefilter("ignore")
            if importer is None:
                module = importlib.import_module(name)
            else:
                module = importer()
        if minimum_version is None or check_min_version(module, minimum_version):
            if callback is not None:
                callback(module, True)
            return module, True
        else:
            version = getattr(module, '__version__', 'UNKNOWN')
            version_error = "version %s does not satisfy the minimum version %s" % (
                version,
                minimum_version,
            )
    except catch_exceptions as e:
        import_error = "%s: %s" % (type(e).__name__, e)

    module = ModuleUnavailable(
        name, error_message, version_error, import_error, package
    )
    if callback is not None:
        callback(module, False)
    return module, False


@deprecated(
    "``declare_deferred_modules_as_importable()`` is deprecated.  "
    "Use the :py:class:`declare_modules_as_importable` context manager.",
    version='6.7.2',
)
def declare_deferred_modules_as_importable(globals_dict):
    """Make all :py:class:`DeferredImportModules` in ``globals_dict`` importable

    This function will go throughout the specified ``globals_dict``
    dictionary and add any instances of :py:class:`DeferredImportModule`
    that it finds (and any of their deferred submodules) to
    ``sys.modules`` so that the modules can be imported through the
    ``globals_dict`` namespace.

    For example, ``pyomo/common/dependencies.py`` declares:

    .. doctest::
       :hide:

       >>> from pyomo.common.dependencies import (
       ...     attempt_import, _finalize_scipy, __dict__ as dep_globals,
       ...     declare_deferred_modules_as_importable, )
       >>> # Sphinx does not provide a proper globals()
       >>> def globals(): return dep_globals

    .. doctest::

       >>> scipy, scipy_available = attempt_import(
       ...     'scipy', callback=_finalize_scipy,
       ...     deferred_submodules=['stats', 'sparse', 'spatial', 'integrate'])
       >>> declare_deferred_modules_as_importable(globals())

    Which enables users to use:

    .. doctest::

       >>> import pyomo.common.dependencies.scipy.sparse as spa

    If the deferred import has not yet been triggered, then the
    :py:class:`DeferredImportModule` is returned and named ``spa``.
    However, if the import has already been triggered, then ``spa`` will
    either be the ``scipy.sparse`` module, or a
    :py:class:`ModuleUnavailable` instance.

    """
    return declare_modules_as_importable(globals_dict).__exit__(None, None, None)


class declare_modules_as_importable(object):
    """Make all :py:class:`ModuleType` and :py:class:`DeferredImportModules`
    importable through the ``globals_dict`` context.

    This context manager will detect all modules imported into the
    specified ``globals_dict`` environment (either directly or through
    :py:func:`attempt_import`) and will make those modules importable
    from the specified ``globals_dict`` context.  It works by detecting
    changes in the specified ``globals_dict`` dictionary and adding any new
    modules or instances of :py:class:`DeferredImportModule` that it
    finds (and any of their deferred submodules) to ``sys.modules`` so
    that the modules can be imported through the ``globals_dict``
    namespace.

    For example, ``pyomo/common/dependencies.py`` declares:

    .. doctest::
       :hide:

       >>> from pyomo.common.dependencies import (
       ...     attempt_import, _finalize_scipy, __dict__ as dep_globals,
       ...     declare_modules_as_importable, )
       >>> # Sphinx does not provide a proper globals()
       >>> def globals(): return dep_globals

    .. doctest::

       >>> with declare_modules_as_importable(globals()):
       ...     scipy, scipy_available = attempt_import(
       ...        'scipy', callback=_finalize_scipy,
       ...        deferred_submodules=['stats', 'sparse', 'spatial', 'integrate'])

    Which enables users to use:

    .. doctest::

       >>> import pyomo.common.dependencies.scipy.sparse as spa

    If the deferred import has not yet been triggered, then the
    :py:class:`DeferredImportModule` is returned and named ``spa``.
    However, if the import has already been triggered, then ``spa`` will
    either be the ``scipy.sparse`` module, or a
    :py:class:`ModuleUnavailable` instance.

    """

    def __init__(self, globals_dict):
        self.globals_dict = globals_dict
        self.init_dict = {}
        self.init_modules = None

    def __enter__(self):
        self.init_dict.update(self.globals_dict)
        self.init_modules = set(sys.modules)

    def __exit__(self, exc_type, exc_value, traceback):
        _global_name = self.globals_dict['__name__'] + '.'
        deferred = {
            k: v
            for k, v in self.globals_dict.items()
            if k not in self.init_dict
            and isinstance(v, (ModuleType, DeferredImportModule))
        }
        if self.init_modules:
            for name in set(sys.modules) - self.init_modules:
                if '.' in name and name.split('.', 1)[0] in deferred:
                    sys.modules[_global_name + name] = sys.modules[name]
        while deferred:
            name, mod = deferred.popitem()
            sys.modules[_global_name + name] = mod
            if isinstance(mod, DeferredImportModule):
                mod.__path__ = None
                mod.__spec__ = None
                deferred.update(
                    (name + '.' + k, v)
                    for k, v in mod.__dict__.items()
                    if type(v) is DeferredImportModule
                )


#
# Common optional dependencies used throughout Pyomo
#

yaml_load_args = {}


def _finalize_yaml(module, available):
    # Recent versions of PyYAML issue warnings if the Loader argument is
    # not set
    if available and hasattr(module, 'SafeLoader'):
        yaml_load_args['Loader'] = module.SafeLoader


def _finalize_ctypes(module, available):
    # ctypes.util must be explicitly imported (and fileutils assumes
    # this has already happened)
    import ctypes.util


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


def _finalize_packaging(module, available):
    if available:
        # Import key subpackages that we will want to assume are present
        import packaging.version


def _finalize_matplotlib(module, available):
    if not available:
        return
    # You must switch matplotlib backends *before* importing pyplot.  If
    # we are in the middle of testing, we need to switch the backend to
    # 'Agg', otherwise attempts to generate plots on CI services without
    # terminal windows will fail.
    if in_testing_environment():
        module.use('Agg')
    import matplotlib.pyplot
    import matplotlib.pylab
    import matplotlib.backends


def _finalize_mpi4py(module, available):
    if not available:
        return
    import mpi4py.MPI


def _finalize_numpy(np, available):
    if not available:
        return
    # scipy has a dependence on numpy.testing, and if we don't import it
    # as part of resolving numpy, then certain deferred scipy imports
    # fail when run under pytest.
    import numpy.testing

    from . import numeric_types

    # Register ndarray as a native type to prevent 1-element ndarrays
    # from accidentally registering ndarray as a native_numeric_type.
    numeric_types.native_types.add(np.ndarray)
    numeric_types.RegisterLogicalType(np.bool_)
    for t in (
        np.int_,
        np.intc,
        np.intp,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        numeric_types.RegisterIntegerType(t)
        # We have deprecated RegisterBooleanType, so we will mock up the
        # registration here (to bypass the deprecation warning) until we
        # finally remove all support for it
        numeric_types._native_boolean_types.add(t)
    _floats = [np.float16, np.float32, np.float64]
    # float96 and float128 may or may not be defined in this particular
    # numpy build (it depends on platform and version).
    # Register them only if they are present
    if hasattr(np, 'float_'):
        # Prepend to preserve previous functionality
        _floats.insert(0, np.float_)
    if hasattr(np, 'float96'):
        _floats.append(np.float96)
    if hasattr(np, 'float128'):
        _floats.append(np.float128)
    for t in _floats:
        numeric_types.RegisterNumericType(t)
        # We have deprecated RegisterBooleanType, so we will mock up the
        # registration here (to bypass the deprecation warning) until we
        # finally remove all support for it
        numeric_types._native_boolean_types.add(t)
    _complex = [np.complex64, np.complex128]
    # complex192 and complex256 may or may not be defined in this
    # particular numpy build (it depends on platform and version).
    # Register them only if they are present
    if hasattr(np, 'np.complex_'):
        # Prepend to preserve functionality
        _complex.insert(0, np.complex_)
    if hasattr(np, 'complex192'):
        _complex.append(np.complex192)
    if hasattr(np, 'complex256'):
        _complex.append(np.complex256)
    for t in _complex:
        numeric_types.RegisterComplexType(t)


def _pyutilib_importer():
    # On newer Pythons, PyUtilib import will fail, but only if a
    # second-level module is imported.  We will arbitrarily choose to
    # check pyutilib.component (as that is the path exercised by the
    # pyomo.common.tempfiles deprecation path)
    importlib.import_module('pyutilib.component')
    return importlib.import_module('pyutilib')


with declare_modules_as_importable(globals()):
    # Standard libraries that are slower to import and not strictly required
    # on all platforms / situations.
    ctypes, _ = attempt_import(
        'ctypes', deferred_submodules=['util'], callback=_finalize_ctypes
    )
    multiprocessing, _ = attempt_import('multiprocessing')
    random, _ = attempt_import('random')

    # Necessary for minimum version checking for other optional dependencies
    packaging, packaging_available = attempt_import(
        'packaging', deferred_submodules=['version'], callback=_finalize_packaging
    )
    # Commonly-used optional dependencies
    dill, dill_available = attempt_import('dill')
    mpi4py, mpi4py_available = attempt_import(
        'mpi4py', deferred_submodules=['MPI'], callback=_finalize_mpi4py
    )
    networkx, networkx_available = attempt_import('networkx')
    numpy, numpy_available = attempt_import('numpy', callback=_finalize_numpy)
    pandas, pandas_available = attempt_import('pandas')
    pathlib, pathlib_available = attempt_import('pathlib')
    pint, pint_available = attempt_import(
        'pint',
        # TypeError for pint<=0.24.3 and python>=3.13
        catch_exceptions=(ImportError, TypeError),
    )
    plotly, plotly_available = attempt_import('plotly')
    pympler, pympler_available = attempt_import('pympler', callback=_finalize_pympler)
    pyutilib, pyutilib_available = attempt_import(
        'pyutilib', importer=_pyutilib_importer
    )
    scipy, scipy_available = attempt_import(
        'scipy',
        callback=_finalize_scipy,
        deferred_submodules=['stats', 'sparse', 'spatial', 'integrate'],
    )
    yaml, yaml_available = attempt_import('yaml', callback=_finalize_yaml)

    # Note that matplotlib.pyplot can generate a runtime error on OSX when
    # not installed as a Framework (as is the case in the CI systems)
    matplotlib, matplotlib_available = attempt_import(
        'matplotlib',
        callback=_finalize_matplotlib,
        deferred_submodules=['pyplot', 'pylab', 'backends'],
        catch_exceptions=(ImportError, RuntimeError),
    )

try:
    import cPickle as pickle
except ImportError:
    import pickle
