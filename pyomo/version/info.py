# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# NOTE: releaselevel should be left undefined (or !='final') for development
#     and set to 'final' for releases.  During development, the
#     major.minor.micro should point to the NEXT release (generally, the
#     next micro release after the current release).
#
# Note: When cutting a release, also update the major/minor/micro in
#
#     pyomo/RELEASE.md
#     .coin-or/projDesc.xml
#
major = 6
minor = 10
micro = 1
# releaselevel = 'final'
serial = 0


def _estimate_release_level():
    "If this is not a final release, attempt to guess the GIT branch or hash"

    from os.path import exists as _exists, join as _join, dirname as _dirname

    if __file__.endswith('setup.py'):
        # This file is being sourced (exec'ed) from setup.py.
        # dirname(__file__) in setup.py's scope is the root source directory
        _rootdir = _dirname(__file__)
    else:
        # Ideally, this should import PYOMO_ROOT_DIR from pyomo.common
        # instead of reimplementing that logic here.  Unfortunately,
        # there is a circular reference (pyomo.common.log imports
        # releaselevel).  We will leave this module completely
        # independent of the rest of Pyomo.
        #
        # __file__ fails if script is called in different ways on Windows
        # __file__ fails if someone does os.chdir() before
        # sys.argv[0] also fails because it doesn't always contain the path
        from inspect import getfile as _getfile, currentframe as _frame
        from os.path import abspath as _abspath

        _rootdir = _join(_dirname(_abspath(_getfile(_frame()))), '..', '..')

    if _exists(_join(_rootdir, '.git')):
        try:
            with open(_join(_rootdir, '.git', 'HEAD')) as _FILE:
                _ref = _FILE.readline().strip()
            releaselevel = 'devel {%s}' % (_ref.split('/')[-1].split('\\')[-1],)
        except:
            releaselevel = 'devel'
    else:
        releaselevel = 'VOTD'

    return releaselevel


def _finalize_version(ver_info):
    "Compute the final version and __version__ strings"

    major, minor, patch, rl, serial = ver_info
    __version__ = f'{major}.{minor}.{patch}'
    if rl.startswith('devel'):
        __version__ += f".dev{serial}"
    elif rl.startswith('VOTD'):
        __version__ += f".a{serial}"

    if rl == 'final':
        version = __version__
    else:
        version = f'{__version__} ({rl})'

    return __version__, version


#
# Set the release level (if this was not marked "final" as part of the
# release process)
#
if globals().get('releaselevel', '') != 'final':
    releaselevel = _estimate_release_level()
#
# Set the version_info, __version__, and version string
#
version_info = (major, minor, micro, releaselevel, serial)
__version__, version = _finalize_version(version_info)
