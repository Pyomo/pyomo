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

from .triangularize import block_triangularize
from .matching import maximum_matching
from .interface import IncidenceGraphInterface, get_bipartite_incidence_graph
from .scc_solver import (
    generate_strongly_connected_components,
    solve_strongly_connected_components,
)
from .incidence import get_incident_variables
from .config import IncidenceMethod

#
# declare deprecation paths for removed modules
#
from pyomo.common.deprecation import moved_module

moved_module(
    "pyomo.contrib.incidence_analysis.util",
    "pyomo.contrib.incidence_analysis.scc_solver",
    version='6.5.0',
    msg=(
        "The 'pyomo.contrib.incidence_analysis.util' module has been moved to "
        "'pyomo.contrib.incidence_analysis.scc_solver'. However, we recommend "
        "importing this functionality (e.g. solve_strongly_connected_components) "
        "directly from 'pyomo.contrib.incidence_analysis'."
    ),
)
del moved_module
