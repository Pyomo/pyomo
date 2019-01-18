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

# The micro number should be set when tagging a release or generating a
# VOTD build. releaselevel should be left at 'invalid' for trunk
# development and set to 'final' for releases.
major=5
minor=6
micro=2
releaselevel='invalid'
#releaselevel='final'
serial=0

if releaselevel == 'final':
    pass
elif '/trunk/' in _init_url or len(_init_url) == 5:
    # __file__ fails if script is called in different ways on Windows
    # __file__ fails if someone does os.chdir() before
    # sys.argv[0] also fails because it doesn't not always contains the path
    from os.path import abspath, dirname, exists, join
    from inspect import getfile, currentframe
    _rootdir = join(dirname(abspath(getfile(currentframe()))), '..', '..')

    if exists(join(_rootdir, '.svn')):
        releaselevel = 'devel {svn}'       #pragma:nocover
    elif exists(join(_rootdir, '.git')):
        try:
            with open(join(_rootdir, '.git', 'HEAD')) as _FILE:
                _ref = _FILE.readline().strip()           #pragma:nocover
            releaselevel = 'devel {%s}' % (
                _ref.split('/')[-1].split('\\')[-1], )    #pragma:nocover
        except:
            releaselevel = 'devel'         #pragma:nocover
    else:
        releaselevel = 'VOTD'              #pragma:nocover

elif '/tags/' in _init_url:                #pragma:nocover
    releaselevel = 'final'

version_info = (major, minor, micro, releaselevel, serial)

if micro:
    version = '.'.join(str(x) for x in version_info[:3])        #pragma:nocover
else:
    version = '.'.join(str(x) for x in version_info[:2])        #pragma:nocover
if releaselevel != 'final':
    version += ' ('+releaselevel+')'

__version__ = '.'.join(str(x) for x in version_info[:3])
