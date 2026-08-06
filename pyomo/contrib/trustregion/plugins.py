# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
# Development of this module was conducted as part of the Institute for
# the Design of Advanced Energy Systems (IDAES) with support through the
# Simulation-Based Engineering, Crosscutting Research Program within the
# U.S. Department of Energy's Office of Fossil Energy and Carbon Management.
# ____________________________________________________________________________________

import logging

logger = logging.getLogger('pyomo.contrib.trustregion')


def load():
    from pyomo.contrib.trustregion.TRF import TrustRegionSolver
