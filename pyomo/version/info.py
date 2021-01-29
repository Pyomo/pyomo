#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

_init_url="$URL$"

# NOTE: releaselevel should be left at 'invalid' for trunk development
#     and set to 'final' for releases.  During development, the
#     major.minor.micro should point ot the NEXT release (generally, the
#     next micro release after the current release).
#
# Note: When cutting a release, also update the major/minor/micro in
#
#     pyomo/RELEASE.txt
#
# The VOTD zipbuilder will automatically change releaselevel to "VOTD
# {hash}" and set the serial number to YYMMDDhhmm.  The serial number
# should generally be left at 0, unless a downstream package is tracking
# master and needs a hard reference to "suitably new" development.
major=5
minor=7
micro=3
#releaselevel='invalid'
releaselevel='final'
serial=0

if releaselevel == 'final':
    pass
elif '/tags/' in _init_url:                #pragma:nocover
    releaselevel = 'final'
elif releaselevel == 'invalid':
    from os.path import abspath, dirname, exists, join
    if __file__.endswith('setup.py'):
        # This file is being sources (exec'ed) from setup.py.
        # dirname(__file__) setup.py's scope is the root sourec directory
        _rootdir = os.path.dirname(__file__)
    else:
        # Eventually this should import PYOMO_ROOT_DIR from
        # pyomo.common instead of reimplementing that logic here.
        #
        # __file__ fails if script is called in different ways on Windows
        # __file__ fails if someone does os.chdir() before
        # sys.argv[0] also fails because it doesn't not always contains the path
        from inspect import getfile, currentframe
        _rootdir = join(dirname(abspath(getfile(currentframe()))), '..', '..')

    if exists(join(_rootdir, '.git')):
        try:
            with open(join(_rootdir, '.git', 'HEAD')) as _FILE:
                _ref = _FILE.readline().strip()           #pragma:nocover
            releaselevel = 'devel {%s}' % (
                _ref.split('/')[-1].split('\\')[-1], )    #pragma:nocover
        except:
            releaselevel = 'devel'         #pragma:nocover
    elif exists(join(_rootdir, '.svn')):
        releaselevel = 'devel {svn}'       #pragma:nocover
    else:
        releaselevel = 'VOTD'              #pragma:nocover


version_info = (major, minor, micro, releaselevel, serial)

version = '.'.join(str(x) for x in version_info[:(3 if micro else 2)])
__version__ = version
if releaselevel != 'final':
    version += ' ('+releaselevel+')'
if releaselevel.startswith('devel'):
    __version__ += ".dev%d" % (serial,)
elif releaselevel.startswith('VOTD'):
    __version__ += "a%d" % (serial,)
