#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from pyomo.common.deprecation import deprecation_warning

try:
    from pysp import *
except ImportError:
    raise ImportError(
        "No module named 'pyomo.pysp'.  "
        "Beginning in Pyomo 6.0, PySP is distributed as a separate "
        "package.  Please see https://github.com/Pyomo/pysp for "
        "information on downloading and installing PySP")

# Redirect all (imported) pysp modules into the pyomo.pysp namespace
import sys
for mod in list(sys.modules):
    if mod.startswith('pysp.'):
        sys.modules['pyomo.'+mod] = sys.modules[mod]
# Warn the user
deprecation_warning("PySP has been removed from pyomo.pysp namespace.  "
                    "Please import PySP directly from the pysp namespace.")
