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
import os
import platform
import six

from . import config

def thisFile():
    """Returns the file name for the module that calls this function.

    This function is more reliable than __file__ on platforms like
    WIndows and in situations where the program has called
    os.chdir().

    """
    # __file__ fails if script is called in different ways on Windows
    # __file__ fails if someone does os.chdir() before
    # sys.argv[0] also fails because it doesn't not always contains the path
    callerFrame = inspect.stack()[1]
    if callerFrame[1] in ('<stdin>','<string>'):
        return callerFrame[1]
    return os.path.abspath(inspect.getfile(callerFrame[0]))


def thisFileDir():
    """Returns the directory containing the module that calls this function.
    """
    return os.path.dirname(thisFile())


PYOMO_ROOT_DIR = os.path.dirname(os.path.dirname(thisFileDir()))


def find_file(fname, cwd=True, mode=os.R_OK, ext=None, pathlist=[]):
    """Locate a file, given a set of search parameters

    Arguments
    ---------
    fname (str): the file name to locate

    cwd (bool): start by looking in the current working directory
        (default: True)

    mode (mask): if not None, only return files that can be accessed for
        reading/writing/executing.  Valid values are the inclusive OR of
        {os.R_OK, os.W_OK, os.X_OK} (default: os.R_OK)

    ext (str): if not None, also look for fname+ext (default: None)

    pathlist (list or str): a list of strings containing paths to
        search, each string contains a single path.  If a string is
        provided, then it is first split using os.pathsep to generate
        the pathlist (default: []).

    Note
    ----
    find_file uses glob, so the path and/or file name may contain
    wildcards.  The first matching file is returned.

    """

    locations = []
    if cwd:
        locations.append(os.getcwd())
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
        for ext in extlist:
            for test in glob.glob(os.path.join(path, fname+ext)):
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
    'windows': '.dll',
    'cygwin':  ('.dll', '.so', '.so.*'),
    'darwin':  '.dynlib',
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

    Arguments
    ---------
    libname (str): the library name to search for

    cwd (bool): Start by looking in the current working directory
        [default: True]

    include_PATH (bool): Include the executable search PATH at the end
        of the list of directories to search. [default: True]

    pathlist (list or str): List of paths to search for the file.  If
        None, then pathlist will default to the local Pyomo
        configuration library directory (and the local Pyomo binary
        directory if include_PATH is set) and the contents of
        LD_LIBRARY_PATH.  If a string, then the string is split using
        os.pathsep.  [Default: None]

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
    extensions (e.g., `.exe`).  Note that as this uses
    :py:func:find_file(), the filename and search paths may contain
    wildcards.

    Arguments
    ---------
    libname (str): the library name to search for

    cwd (bool): Start by looking in the current working directory
        [default: True]

    include_PATH (bool): Include the executable search PATH at the end
        of the list of directories to search. [default: True]

    pathlist (list or str): List of paths to search for the file.  If
        None, then pathlist will default to the local Pyomo
        configuration binary directory.  If a string, then the string is
        split using os.pathsep.  [Default: None]

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
                     pathlist=pathlist)
