#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""Pyomo: Python Optimization Modeling Objects

Pyomo provides Python packages for formulating and solving complex
optimization applications.  Most of Pyomo's packages rely on externally
built optimization solvers.

pyomo.version provides a mechanism for managing stuff that is related to
releases of the entire Pyomo software.
"""

_init_url="$URL$"

# The micro number should be set when tagging a release or generating a
# VOTD build
_major=5
_minor=1
_micro=0
_releaselevel='final'
_serial=0

if '/trunk/' in _init_url or len(_init_url) == 5:
    # __file__ fails if script is called in different ways on Windows
    # __file__ fails if someone does os.chdir() before
    # sys.argv[0] also fails because it doesn't not always contains the path
    from os.path import abspath, dirname, exists, join
    from inspect import getfile, currentframe
    _rootdir = join(dirname(abspath(getfile(currentframe()))), '..', '..')

    if exists(join(_rootdir, '.svn')):
        _releaselevel = 'trunk'             #pragma:nocover
    elif exists(join(_rootdir, '.git')):
        _releaselevel = 'trunk {git}'       #pragma:nocover
    else:
        _releaselevel = 'VOTD'              #pragma:nocover

elif '/tags/' in _init_url:                 #pragma:nocover
    _releaselevel = 'final'

version_info = (_major, _minor, _micro, _releaselevel, _serial)

if _micro:
    version = '.'.join(str(x) for x in version_info[:3])        #pragma:nocover
else:
    version = '.'.join(str(x) for x in version_info[:2])        #pragma:nocover
if _releaselevel != 'final':
    version += ' ('+_releaselevel+')'
