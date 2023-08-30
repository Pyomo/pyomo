#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import platform

if 'PYOMO_CONFIG_DIR' in os.environ:
    PYOMO_CONFIG_DIR = os.path.abspath(os.environ['PYOMO_CONFIG_DIR'])
elif platform.system().lower().startswith(('windows', 'cygwin')):
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
