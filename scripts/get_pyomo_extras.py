#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# A script to optionally install packages that Pyomo could leverage.
#

import inspect
import os
import sys

from os.path import dirname, abspath

#
# THIS SCRIPT IS DEPRECATED.  Please use "pyomo install-extras"
#

if __name__ == '__main__':
    try:
        callerFrame = inspect.stack()[0]
        _dir = os.path.join(
            dirname(dirname(abspath(inspect.getfile(callerFrame[0])))),
            'pyomo',
            'scripting',
            'plugins',
        )
        sys.path.insert(0, _dir)
        extras = __import__('extras')
        extras.install_extras()
    except:
        print("Error running get-pyomo-extras.py")
