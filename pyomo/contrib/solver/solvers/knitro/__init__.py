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

from pyomo.contrib.solver.common.factory import SolverFactory

from .config import KnitroConfig
from .direct import KnitroDirectSolver

__all__ = ["KnitroConfig", "KnitroDirectSolver"]


# This function needs to be called from the plugins load function
def load():
    SolverFactory.register(
        name="knitro_direct",
        legacy_name="knitro_direct",
        doc="Direct interface to KNITRO solver",
    )(KnitroDirectSolver)
