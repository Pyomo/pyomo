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

_init_url = "$URL$"

# NOTE: releaselevel should be left at 'invalid' for trunk development
#     and set to 'final' for releases.  During development, the
#     major.minor.micro should point to the NEXT release (generally, the
#     next micro release after the current release).
#
# Note: When cutting a release, also update the major/minor/micro in
#
#     pyomo/RELEASE.md
#
# The VOTD zipbuilder will automatically change releaselevel to "VOTD
# {hash}" and set the serial number to YYMMDDhhmm.  The serial number
# should generally be left at 0, unless a downstream package is tracking
# main and needs a hard reference to "suitably new" development.
major = 6
minor = 9
micro = 4
# releaselevel = 'invalid'
releaselevel = 'final'
serial = 0

if releaselevel == 'final':
    pass
elif '/tags/' in _init_url:  # pragma:nocover
    releaselevel = 'final'
elif releaselevel == 'invalid':
    from os.path import exists as _exists, join as _join, dirname as _dirname

    if __file__.endswith('setup.py'):
        # This file is being sourced (exec'ed) from setup.py.
        # dirname(__file__) setup.py's scope is the root source directory
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


version_info = (major, minor, micro, releaselevel, serial)

__version__ = '.'.join(str(x) for x in version_info[:3])
if releaselevel.startswith('devel'):
    __version__ += ".dev%d" % (serial,)
elif releaselevel.startswith('VOTD'):
    __version__ += "a%d" % (serial,)

version = __version__
if releaselevel != 'final':
    version += ' (' + releaselevel + ')'
