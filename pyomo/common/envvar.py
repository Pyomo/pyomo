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

"""Attributes describing the current platform and user configuration.

This module provides standardized attributes that other parts of Pyomo
can use to interrogate aspects of the current platform, and to find
information about the current user configuration (including where to
locate the main Pyomo configuration directory).

"""

import os
import platform

_platform = platform.system().lower()
#: bool : True if running in a "native" Windows environment
is_native_windows = _platform.startswith('windows')
#: bool : True if running on Windows (native or cygwin)
is_windows = is_native_windows or _platform.startswith('cygwin')
#: bool : True if running on Mac/OSX
is_osx = _platform.startswith('darwin')
#: bool: True if running under the PyPy interpreter
is_pypy = platform.python_implementation().lower().startswith('pypy')

#: str : Absolute path to the user's Pyomo Configuration Directory.
#:
#:     By default, this is ``~/.pyomo`` on Linux and OSX and
#:     ``%LOCALAPPDATA%/Pyomo`` on Windows.  It can be overridden by
#:     setting the ``PYOMO_CONFIG_DIR`` environment variable before
#:     importing Pyomo.
PYOMO_CONFIG_DIR = None

if 'PYOMO_CONFIG_DIR' in os.environ:
    PYOMO_CONFIG_DIR = os.path.abspath(os.environ['PYOMO_CONFIG_DIR'])
elif is_windows:
    PYOMO_CONFIG_DIR = os.path.abspath(
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Pyomo')
    )
else:
    PYOMO_CONFIG_DIR = os.path.abspath(
        os.path.join(os.environ.get('HOME', ''), '.pyomo')
    )

# Note that alternative platform-independent implementation of the above
# could be to use:
#
#   PYOMO_CONFIG_DIR = os.path.abspath(appdirs.user_data_dir('pyomo'))
#
# But would require re-adding the hard dependency on appdirs.  For now
# (13 Jul 20), the above appears to be sufficiently robust.
