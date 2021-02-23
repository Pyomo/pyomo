#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import ctypes.util
import glob
import inspect
import logging
import os
import platform
import six

from .deprecation import deprecated
from . import config

def this_file(stack_offset=1):
    """Returns the file name for the module that calls this function.

    This function is more reliable than __file__ on platforms like
    Windows and in situations where the program has called
    os.chdir().

    """
    # __file__ fails if script is called in different ways on Windows
    # __file__ fails if someone does os.chdir() before
    # sys.argv[0] also fails because it does not always contains the path
    callerFrame = inspect.currentframe()
    while stack_offset:
        callerFrame = callerFrame.f_back
        stack_offset -= 1
    frameName = callerFrame.f_code.co_filename
    if frameName and frameName[0] == '<' and frameName[-1] == '>':
        return frameName
    return os.path.abspath(inspect.getfile(callerFrame))


def this_file_dir():
    """Returns the directory containing the module that calls this function.
    """
    return os.path.dirname(this_file(stack_offset=2))


PYOMO_ROOT_DIR = os.path.dirname(os.path.dirname(this_file_dir()))


def find_path(name, validate, cwd=True, mode=os.R_OK, ext=None,
              pathlist=[], allow_pathlist_deep_references=True):
    """Locate a path, given a set of search parameters

    Parameters
    ----------
    name : str
        The name to locate.  The name may contain references to a user's
        home directory (``~user``), environment variables
        (``${HOME}/bin``), and shell wildcards (``?`` and ``*``); all of
        which will be expanded.

    validate : function
        A function to call to validate the path (used by find_file and
        find_dir to discriminate files and directories)

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    mode : mask
        If not None, only return entries that can be accessed for
        reading/writing/executing.  Valid values are the inclusive OR of
        {os.R_OK, os.W_OK, os.X_OK} [default: ``os.R_OK``]

    ext : str or iterable of str
        If not None, also look for name+ext [default: None]

    pathlist : str or iterable of str
        A list of strings containing paths to search, each string
        contains a single path.  If pathlist is a string, then it is
        first split using os.pathsep to generate the pathlist
        [default: ``[]``].

    allow_pathlist_deep_references : bool
       If allow_pathlist_deep_references is True and the name
       appears to be a relative path, allow deep reference matches
       relative to directories in the pathlist (e.g., if name is
       ``foo/my.exe`` and ``/usr/bin`` is in the pathlist, then
       :py:func:`find_file` could return ``/usr/bin/foo/my.exe``).  If
       allow_pathlist_deep_references is False and the name appears
       to be a relative path, then only matches relative to the current
       directory are allowed (assuming cwd==True).  [default: True]

    Notes
    -----
        find_path uses glob, so the path and/or name may contain
        wildcards.  The first matching entry is returned.

    """

    # Support shell-style paths like ~user and $HOME/bin
    name = os.path.expanduser(os.path.expandvars(name))

    locations = []
    if cwd:
        locations.append(os.getcwd())

    if allow_pathlist_deep_references or os.path.basename(name) == name:
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
            for test in glob.glob(os.path.join(path, name+_ext)):
                if not validate(test):
                    continue
                if mode is not None and not os.access(test, mode):
                    continue
                return os.path.abspath(test)
    return None


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
    return find_path(
        filename, os.path.isfile, cwd=cwd, mode=mode, ext=ext,
        pathlist=pathlist,
        allow_pathlist_deep_references=allow_pathlist_deep_references
    )



def find_dir(dirname, cwd=True, mode=os.R_OK, pathlist=[],
             allow_pathlist_deep_references=True):
    """Locate a directory, given a set of search parameters

    Parameters
    ----------
    dirname : str
        The directory name to locate.  The name may contain references
        to a user's home directory (``~user``), environment variables
        (``${HOME}/bin``), and shell wildcards (``?`` and ``*``); all of
        which will be expanded.

    cwd : bool
        Start by looking in the current working directory
        [default: True]

    mode : mask
        If not None, only return directories that can be accessed for
        reading/writing/executing.  Valid values are the inclusive OR of
        {os.R_OK, os.W_OK, os.X_OK} [default: ``os.R_OK``]

    pathlist : str or iterable of str
        A list of strings containing paths to search, each string
        contains a single path.  If pathlist is a string, then it is
        first split using os.pathsep to generate the pathlist
        [default: ``[]``].

    allow_pathlist_deep_references : bool
       If allow_pathlist_deep_references is True and the dirname
       appears to be a relative path, allow deep reference matches
       relative to directories in the pathlist (e.g., if dirname is
       ``foo/bar`` and ``/usr/bin`` is in the pathlist, then
       :py:func:`find_dir` could return ``/usr/bin/foo/bar``).  If
       allow_pathlist_deep_references is False and the dirname appears
       to be a relative path, then only matches relative to the current
       directory are allowed (assuming cwd==True).  [default: True]

    Notes
    -----
        find_dir uses glob, so the path and/or directory name may contain
        wildcards.  The first matching directory is returned.

    """
    return find_path(
        dirname, os.path.isdir, cwd=cwd, mode=mode, pathlist=pathlist,
        allow_pathlist_deep_references=allow_pathlist_deep_references
    )


_exeExt = {
    'linux':   None,
    'windows': '.exe',
    'cygwin':  '.exe',
    'darwin':  None,
}

_libExt = {
    'linux':   ('.so', '.so.*'),
    'windows': ('.dll', '.pyd'),
    'cygwin':  ('.dll', '.so', '.so.*'),
    'darwin':  ('.dylib', '.so', '.so.*'),
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

    If the explicit path search fails to locate a library, then this
    returns the result from passing the basename (with 'lib' and extension
    removed) to ctypes.util.find_library()

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
    # Search 1: original filename (with extensions) in our paths
    lib = find_file(libname, cwd=cwd, ext=ext, pathlist=pathlist)
    if lib is None and not libname.startswith('lib'):
        # Search 2: prepend 'lib' (with extensions) in our paths
        lib = find_file('lib'+libname, cwd=cwd, ext=ext, pathlist=pathlist)
    if lib is not None:
        return lib
    # Search 3: use ctypes.util.find_library (which expects 'lib' and
    # extension to be removed from the name)
    if libname.startswith('lib') and _system() != 'windows':
        libname = libname[3:]
    libname_base, ext = os.path.splitext(os.path.basename(libname))
    if ext.lower().startswith(('.so','.dll','.dylib')):
        return ctypes.util.find_library(libname_base)
    else:
        return ctypes.util.find_library(libname)


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


class _PathData(object):
    def __init__(self, manager, name):
        self._mngr = manager
        self._registered_name = name
        self._path = None
        self._path_override = None
        self._status = None

    def path(self):
        """Return the full, normalized path to the registered path entry.

        If the object is not found (or was marked "disabled"),
        ``path()`` returns None.

        """
        if self._status is None:
            if self._path_override:
                target = self._path_override
            else:
                target = self._registered_name
            tmp = self._mngr._find(target, pathlist=self._mngr.pathlist)
            self._path = tmp if tmp else self._path_override
            self._status = bool(tmp)
        return self._path

    def set_path(self, value):
        self._path_override = value
        self.rehash()
        if not self._status:
            logging.getLogger('pyomo.common').warning(
                "explicitly setting the path for '%s' to an "
                "invalid object or nonexistent location ('%s')"
                % (self._registered_name, value))

    @deprecated("get_path() is deprecated; use "
                "pyomo.common.Executable(name).path()",
                version='5.6.2')
    def get_path(self):
        return self.path()

    def disable(self):
        """Disable this path entry

        This method "disables" this path entry by marking it as "not
        found".  Disabled entries return False for `available()` and
        None for `path()`.  The disabled status will persist until the
        next call to `rehash()`.

        """
        self._status = False
        self._path = None

    def available(self):
        """Returns True if the registered path is available.

        Entries are available if the object was found found in the
        search locations and has not been explicitly disabled.

        """
        if self._status is None:
            self.path()
        return self._status

    def rehash(self):
        """Requery the location of this path entry

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


class _ExecutableData(_PathData):
    @property
    def executable(self):
        return self.path()

    @executable.setter
    def executable(self, value):
        self.set_path(value)


class PathManager(object):
    """The PathManager defines a registry class for path locations

    The :py:class:`PathManager` defines a class very similar to the
    :py:class:`CachedFactory` class; however it does not register type
    constructors.  Instead, it registers instances of
    :py:class:`_PathData` (or :py:class:`_ExecutableData`).  These
    contain the resolved path to the directory object under which the
    :py:class:`_PathData` object was registered.  We do not use
    the PyUtilib ``register_executable`` and ``registered_executable``
    functions so that we can automatically include Pyomo-specific
    locations in the search path (namely the ``PYOMO_CONFIG_DIR``).

    Users will generally interact with this class through global
    instances of this class (``pyomo.common.Executable`` and
    ``pyomo.common.Library``).

    Users are not required or expected to register file names with the
    :py:class:`PathManager`; they will be automatically registered
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
    available by casting the :py:class:`_PathData` object requrned
    from ``Executable`` or ``Library`` to either a ``bool`` or ``str``:

    .. doctest::

        >>> if Executable('demo_exec_file'):
        ...     cmd = "%s --help" % Executable('demo_exec_file')

    The :py:class:`PathManager` caches the location / existence of
    the target directory entry.  If something in the environment changes
    (e.g., the PATH) or the file is created or removed after the first
    time a client queried the location or availability, the
    PathManager will return incorrect information.  You can cause
    the :py:class:`PathManager` to refresh its cache by calling
    ``rehash()`` on either the :py:class:`_PathData` (for the
    single file) or the :py:class:`PathManager` to refresh the
    cache for all files:

    .. doctest::

        >>> # refresh the cache for a single file
        >>> Executable('demo_exec_file').rehash()
        >>> # or all registered files
        >>> Executable.rehash()

    The ``Executable`` singleton looks for executables in the system
    ``PATH`` and in the list of directories specified by the ``pathlist``
    attribute.  ``Executable.pathlist`` defaults to a list containing the
    ``os.path.join(pyomo.common.config.PYOMO_CONFIG_DIR, 'bin')``.

    The ``Library`` singleton looks for executables in the system
    ``LD_LIBRARY_PATH``, ``PATH`` and in the list of directories
    specified by the ``pathlist`` attribute.  ``Library.pathlist``
    defaults to a list containing the
    ``os.path.join(pyomo.common.config.PYOMO_CONFIG_DIR, 'lib')``.

    Users may also override the normal file resolution by explicitly
    setting the location using :py:meth:`set_path`:

    .. doctest::

        >>> Executable('demo_exec_file').set_path(os.path.join(
        ...     pyomo.common.config.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file'))

    Explicitly setting the path is an absolute operation and will
    set the location whether or not that location points to an actual
    file.  Additionally, the explicit location will persist
    through calls to ``rehash()``.  If you wish to remove the explicit
    executable location, call ``set_path(None)``:

    .. doctest::

        >>> Executable('demo_exec_file').set_path(None)

    The ``Executable`` singleton uses :py:class:`_ExecutableData`, an
    extended form of the :py:class:`_PathData` class, which provides the
    ``executable`` property as an alais for :py:meth:`path()` and
    :py:meth:`set_path()`:

    .. doctest::

        >>> loc = Executable('demo_exec_file').executable
        >>> print(os.path.isfile(loc))
        >>> Executable('demo_exec_file').executable = os.path.join(
        ...     pyomo.common.config.PYOMO_CONFIG_DIR, 'bin', 'demo_exec_file')
        >>> Executable('demo_exec_file').executable = None

    .. doctest::
        :hide:

        >>> if _del_testfile:
        ...     os.remove(_testfile)

    """
    def __init__(self, finder, dataClass):
        self._pathTo = {}
        self._find = finder
        self._dataClass = dataClass
        self.pathlist = None

    def __call__(self, path):
        if path not in self._pathTo:
            self._pathTo[path] = self._dataClass(self, path)
        return self._pathTo[path]

    def rehash(self):
        """Requery the location of all registered executables

        This method derives its name from the csh command of the same
        name, which rebuilds the hash table of executables reachable
        through the PATH.

        """
        for _path in six.itervalues(self._pathTo):
            _path.rehash()

#
# Define singleton objects for Pyomo / Users to interact with
#
Executable = PathManager(find_executable, _ExecutableData)
Library = PathManager(find_library, _PathData)


@deprecated("pyomo.common.register_executable(name) has been deprecated; "
            "explicit registration is no longer necessary",
            version='5.6.2')
def register_executable(name, validate=None):
    # Setting to None will cause Executable to re-search the pathlist
    return Executable(name).rehash()

@deprecated(
    """pyomo.common.registered_executable(name) has been deprecated; use
    pyomo.common.Executable(name).path() to get the path or
    pyomo.common.Executable(name).available() to get a bool indicating
    file availability.  Equivalent results can be obtained by casting
    Executable(name) to string or bool.""",
    version='5.6.2')
def registered_executable(name):
    ans = Executable(name)
    if ans.path() is None:
        return None
    else:
        return ans

@deprecated("pyomo.common.unregister_executable(name) has been deprecated; "
            "use Executable(name).disable()",
            version='5.6.2')
def unregister_executable(name):
    Executable(name).disable()
