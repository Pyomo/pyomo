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

import logging
import sys
from pyomo.common.deprecation import deprecation_warning, in_testing_environment

try:
    # Warn the user
    deprecation_warning(
        "PySP has been removed from the pyomo.pysp namespace.  "
        "Please import PySP directly from the pysp namespace.",
        version='6.0',
    )
    from pysp import *

    # Redirect all (imported) pysp modules into the pyomo.pysp namespace
    for mod in list(sys.modules):
        if mod.startswith('pysp.'):
            sys.modules['pyomo.' + mod] = sys.modules[mod]
except ImportError:
    # Only raise the exception if nose/pytest/sphinx are NOT running
    # (otherwise test discovery can result in exceptions)
    if not in_testing_environment():
        raise ImportError(
            "No module named 'pyomo.pysp'.  "
            "Beginning in Pyomo 6.0, PySP is distributed as a separate "
            "package.  Please see https://github.com/Pyomo/pysp for "
            "information on downloading and installing PySP"
        )
