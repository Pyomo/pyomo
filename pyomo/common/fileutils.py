#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import glob
import inspect
import logging
import os
import platform
import six

from .deprecation import deprecated
from . import config

def this_file():
    """Returns the file name for the module that calls this function.

    This function is more reliable than __file__ on platforms like
    WIndows and in situations where the program has called
    os.chdir().

    """
    # __file__ fails if script is called in different ways on Windows
    # __file__ fails if someone does os.chdir() before
    # sys.argv[0] also fails because it does not always contains the path
    callerFrame = inspect.stack()[1]
    frameName = callerFrame[1]
    if frameName and frameName[0] == '<' and frameName[-1] == '>':
        return frameName
    return os.path.abspath(inspect.getfile(callerFrame[0]))


def this_file_dir():
    """Returns the directory containing the module that calls this function.
    """
    return os.path.dirname(this_file())


PYOMO_ROOT_DIR = os.path.dirname(os.path.dirname(this_file_dir()))


def find_file(filename, cwd=True, mode=os.R_OK, ext=None, pathlist=[],
              allow_pathlist_deep_references=True):
    """Locate a file, given a set of search parameters

    Parameters
    ----------
    filename : str
        The file name to locate.  The file name may contain references
        to a user's home directory (``~user``), environment variables
        (``${HOME}/bin``), and shell wildcards (``?`` and ``*``); all of
        which will be expanded.

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    mode : mask
        If not None, only return files that can be accessed for
        reading/writing/executing.  Valid values are the inclusive OR of
        {os.R_OK, os.W_OK, os.X_OK} [default: ``os.R_OK``]

    ext : str or iterable of str
        If not None, also look for filename+ext [default: None]

    pathlist : str or iterable of str
        A list of strings containing paths to search, each string
        contains a single path.  If pathlist is a string, then it is
        first split using os.pathsep to generate the pathlist
        [default: ``[]``].

    allow_pathlist_deep_references : bool
       If allow_pathlist_deep_references is True and the filename
       appears to be a relative path, allow deep reference matches
       relative to directories in the pathlist (e.g., if filename is
       ``foo/my.exe`` and ``/usr/bin`` is in the pathlist, then
       :py:func:`find_file` could return ``/usr/bin/foo/my.exe``).  If
       allow_pathlist_deep_references is False and the filename appears
       to be a relative path, then only matches relative to the current
       directory are allowed (assuming cwd==True).  [default: True]

    Notes
    -----
        find_file uses glob, so the path and/or file name may contain
        wildcards.  The first matching file is returned.

    """

    # Support shell-style paths like ~user and $HOME/bin
    filename = os.path.expanduser(os.path.expandvars(filename))

    locations = []
    if cwd:
        locations.append(os.getcwd())

    if allow_pathlist_deep_references or os.path.basename(filename) == filename:
        if isinstance(pathlist, six.string_types):
            locations.extend( pathlist.split(os.pathsep) )
        else:
            locations.extend(pathlist)

    extlist = ['']
    if ext:
        if isinstance(ext, six.string_types):
            extlist.append(ext)
        else:
            extlist.extend(ext)

    for path in locations:
        if not path:
            continue
        for _ext in extlist:
            for test in glob.glob(os.path.join(path, filename+_ext)):
                if not os.path.isfile(test):
                    continue
                if mode is not None and not os.access(test, mode):
                    continue
                return os.path.abspath(test)
    return None

_exeExt = {
    'linux':   None,
    'windows': '.exe',
    'cygwin':  '.exe',
    'darwin':  None,
}

_libExt = {
    'linux':   ('.so', '.so.*'),
    'windows': ('.dll',),
    'cygwin':  ('.dll', '.so', '.so.*'),
    'darwin':  ('.dylib',),
}

def _system():
    system = platform.system().lower()
    for c in '.-_':
        system = system.split(c)[0]
    return system


def _path():
    return (os.environ.get('PATH','') or os.defpath).split(os.pathsep)


def find_library(libname, cwd=True, include_PATH=True, pathlist=None):
    """Find a dynamic library using find_file to search typical locations.

    Finds a specified library (file) by searching a specified set of
    paths.  This routine will look for the specified file name, as well
    as looking for the filename followed by architecture-specific
    extensions (e.g., `.dll`, `.so`, or `.dynlib`).  Note that as this
    uses :py:func:find_file(), the filename and search paths may contain
    wildcards.

    Parameters
    ----------
    libname : str
        The library name to search for

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    include_PATH : bool
        Include the executable search PATH at the end of the list of
        directories to search. [default: True]

    pathlist : str or list of str
        List of paths to search for the file.  If None, then pathlist
        will default to the local Pyomo configuration library directory
        (and the local Pyomo binary directory if include_PATH is set)
        and the contents of LD_LIBRARY_PATH.  If a string, then the
        string is split using os.pathsep.  [default: None]

    Notes
    -----
        find_library() uses :py:func:`find_file` with
        ``allow_pathlist_deep_references=True``, so libnames containing
        relative paths will be matched relative to all paths in
        pathlist.
    """
    if pathlist is None:
        # Note: PYOMO_CONFIG_DIR/lib comes before LD_LIBRARY_PATH, and
        # PYOMO_CONFIG_DIR/bin comes immediately before PATH
        pathlist = [ os.path.join(config.PYOMO_CONFIG_DIR, 'lib') ]
        pathlist.extend(os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep))
        if include_PATH:
            pathlist.append( os.path.join(config.PYOMO_CONFIG_DIR, 'bin') )
    elif isinstance(pathlist, six.string_types):
        pathlist = pathlist.split(os.pathsep)
    else:
        pathlist = list(pathlist)
    if include_PATH:
        pathlist.extend(_path())
    ext = _libExt.get(_system(), None)
    return find_file(libname, cwd=cwd, ext=ext, pathlist=pathlist)


def find_executable(exename, cwd=True, include_PATH=True, pathlist=None):
    """Find an executable using find_file to search typical locations.

    Finds a specified executable by searching a specified set of paths.
    This routine will look for the specified file name, as well as
    looking for the filename followed by architecture-specific
    extensions (e.g., ``.exe``).  Note that as this uses
    :py:func:`find_file()`, the filename and search paths may contain
    wildcards.

    Parameters
    ----------
    exename : str
        The executable file name to search for

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    include_PATH : bool
        Include the executable search PATH at the end of the list of
        directories to search. [default: True]

    pathlist : str or list of str
        List of paths to search for the file.  If None, then pathlist
        will default to the local Pyomo configuration binary directory.
        If a string, then the string is split using os.pathsep.
        [Default: None]

    Notes
    -----
        find_executable() uses :py:func:`find_file` with
        ``allow_pathlist_deep_references=False``, so search strings
        containing relative paths will only be matched relative to the
        current working directory.  This prevents confusion in the case
        where a user called ``find_executable("./foo")`` and forgot to copy
        ``foo`` into the local directory, but this function picked up
        another ``foo`` in the user's PATH that they did not want to use.

    """
    if pathlist is None:
        pathlist = [ os.path.join(config.PYOMO_CONFIG_DIR, 'bin') ]
    elif isinstance(pathlist, six.string_types):
        pathlist = pathlist.split(os.pathsep)
    else:
        pathlist = list(pathlist)
    if include_PATH:
        pathlist.extend(_path())
    ext = _exeExt.get(_system(), None)
    return find_file(exename, cwd=cwd, ext=ext, mode=os.R_OK|os.X_OK,
                     pathlist=pathlist, allow_pathlist_deep_references=False)


class _ExecutableData(object):
    def __init__(self, manager, executable):
        self._mngr = manager
        self._registered_name = executable
        self._path = None
        self._path_override = None
        self._status = None

    def path(self):
        """Return the full, normalized path to the registered executable.

        If the executable is not found (or was marked "disabled"),
        ``path()`` returns None.

        """
        if self._status is None:
            if self._path_override:
                target = self._path_override
            else:
                target = self._registered_name
            tmp = find_executable(target, pathlist=self._mngr.pathlist)
            self._path = tmp if tmp else self._path_override
            self._status = bool(tmp)
        return self._path

    @deprecated("registered_executable(name).get_path() is deprecated; "
                "use pyomo.common.Executable(name).path()")
    def get_path(self):
        return self.path()

    def disable(self):
        """Disable this executable

        This method "disables" this executable by marking it as "not
        found".  Disabled executables return False for `available()` and
        None for `path()`.  The disabled status will persist until the
        next call to `rehash()`.

        """
        self._status = False
        self._path = None

    def available(self):
        """Return True if the executable was found in the search locations"""
        if self._status is None:
            self.path()
        return self._status

    def rehash(self):
        """Requery the location of this executables

        This method derives its name from the csh command of the same
        name, which rebuilds the hash table of executables reachable
        through the PATH.

        """
        self._status = None
        self.path()

    def __nonzero__(self):
        """Alias for ``available()``."""
        return self.available()

    # "if <obj>" triggers __nonzero__ in Py2 and __bool__ in Py3
    __bool__ = __nonzero__

    def __str__(self):
        ans = self.path()
        if not ans:
            return ""
        return ans

    @property
    def executable(self):
        return self.path()

    @executable.setter
    def executable(self, value):
        self._path_override = value
        self.rehash()
        if not self._status:
            logging.getLogger('pyomo.common').warning(
                "explicitly setting the path for executable '%s' to a "
                "non-executable file or nonexistent location ('%s')"
                % (self._registered_name, value))


class ExecutableManager(object):
    """The ExecutableManager defines a registry class for executable files

    The :py:class:`ExecutableManager` defines a class very similar to the
    :py:class:`CachedFactory` class; however it does not register type
    constructors.  Instead, it registers instances of
    :py:class:`_ExecutableData`.  These contain the resolved path to the
    executable under which the :py:class:`_ExecutableData` object was
    registered.  We do not use the PyUtilib ``register_executable`` and
    ``registered_executable`` functions so that we can automatically
    include Pyomo-specific locations in the search path (namely the
    ``PYOMO_CONFIG_DIR``).

    Pyomo declares a single global instance of this class as
    ``pyomo.common.Executable``.

    Users are not required or expected to register file names with the
    :py:class:`ExecutableManager`; they will be automatically registered
    upon first use.  Generally, users interact through the ``path()``
    and ``available()`` methods:

    .. doctest::
        :hide:

        >>> import pyomo.common
        >>> import os
        >>> from stat import S_IXUSR, S_IXGRP, S_IXOTH
        >>> _testfile = os.path.join(
        ...    pyomo.common.config.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file')
        >>> _del_testfile = not os.path.exists(_testfile)
        >>> if _del_testfile:
        ...     open(_testfile,'w').close()
        ...     _mode = os.stat(_testfile).st_mode
        ...     os.chmod(_testfile, _mode | S_IXUSR | S_IXGRP | S_IXOTH)

    .. doctest::

        >>> from pyomo.common import Executable
        >>> if Executable('demo_exec_file').available():
        ...     loc = Executable('demo_exec_file').path()
        ...     print(os.path.isfile(loc))
        True
        >>> print(os.access(loc, os.X_OK))
        True

    For convenience, :py:meth:`available()` and :py:meth:`path()` are
    available by casting the :py:class:`_ExecutableData` object requrned
    from ``Executable`` to either a ``bool`` or ``str``:

    .. doctest::

        >>> if Executable('demo_exec_file'):
        ...     cmd = "%s --help" % Executable('demo_exec_file')

    The :py:class:`ExecutableManager` caches the location / existence of
    the target executable.  If something in the environment changes
    (e.g., the PATH) or the file is created or removed after the first
    time a client queried the location or availability, the
    ExecutionManager will return incorrect information.  You can cause
    the :py:class:`ExecutionManager` to refresh its cache by calling
    ``rehash()`` on either the :py:class:`_ExecutableData` (for the
    single file) or the :py:class:`ExecutionManager` to refresh the
    cache for all files:

    .. doctest::

        >>> # refresh the cache for a single file
        >>> Executable('demo_exec_file').rehash()
        >>> # or all registered files
        >>> Executable.rehash()

    :py:class:`ExecutionManager` looks for executables in the system
    `PATH` and in the list of directories specified by the ``pathlist``
    attribute.  ``Executable.pathlist`` defaults to a list containing the
    initial value of `pyomo.common.config.PYOMO_CONFIG_DIR`.

    Users may also override the normal file resolution by explicitly
    setting the files ``executable`` attribute:

    .. doctest::

        >>> Executable('demo_exec_file').executable = os.path.join(
        ...     pyomo.common.config.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file')

    Explicitly setting the executable is an absolute operation and will
    set the location whether or not that location points to an actual
    executable file.  Additionally, the explicit location will persist
    through calls to ``rehash()``.  If you wish to remove the explicit
    executable location, set the ``executable`` to ``None``:

    .. doctest::

        >>> Executable('demo_exec_file').executable = None


    .. doctest::
        :hide:

        >>> if _del_testfile:
        ...     os.remove(_testfile)

    """
    def __init__(self):
        self._exec = {}
        self.pathlist = [ os.path.join(config.PYOMO_CONFIG_DIR, 'bin') ]

    def __call__(self, executable):
        if executable not in self._exec:
            self._exec[executable] = _ExecutableData(self, executable)
        return self._exec[executable]

    def rehash(self):
        """Requery the location of all registered executables

        This method derives its name from the csh command of the same
        name, which rebuilds the hash table of executables reachable
        through the PATH.

        """
        for _exe in six.itervalues(self._exec):
            _exe.rehash()

Executable = ExecutableManager()

@deprecated("pyomo.common.register_executable(name) has been deprecated; "
            "explicit registration is no longer necessary")
def register_executable(name, validate=None):
    # Setting to None will cause Executable to re-search the pathlist
    return Executable(name).rehash()

@deprecated(
    """pyomo.common.registered_executable(name) has been deprecated; use
    pyomo.common.Executable(name).path() to get the path or
    pyomo.common.Executable(name).available() to get a bool indicating
    file availability.  Equivalent results can be obtained by casting
    Executable(name) to string or bool.""")
def registered_executable(name):
    ans = Executable(name)
    if ans.path() is None:
        return None
    else:
        return ans

@deprecated("pyomo.common.unregister_executable(name) has been deprecated; "
            "use Executable(name).disable()")
def unregister_executable(name):
    Executable(name).disable()
