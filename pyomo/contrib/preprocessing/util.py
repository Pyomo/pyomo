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

import logging

from io import StringIO

from pyomo.common.log import LoggingIntercept


class SuppressConstantObjectiveWarning(LoggingIntercept):
    """Suppress the infeasible model warning message from solve().

    The "WARNING: Constant objective detected, replacing with a placeholder"
    warning message from calling solve() is often unwanted, but there is no
    clear way to suppress it.

    TODO need to fix this so that it only suppresses the desired message.

    """

    def __init__(self):
        super(SuppressConstantObjectiveWarning, self).__init__(
            StringIO(), 'pyomo.core', logging.WARNING
        )
