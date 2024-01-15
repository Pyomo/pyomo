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

from pyomo.common.deprecation import relocated_module

msg = (
    "The 'pyomo.contrib.incidence_analysis.util' module has been moved to"
    " 'pyomo.contrib.incidence_analysis.scc_solver'. However, we recommend"
    " importing this functionality (e.g. solve_strongly_connected_components)"
    " directly from 'pyomo.contrib.incidence_analysis'."
)
relocated_module(
    "pyomo.contrib.incidence_analysis.scc_solver", version='6.5.0', msg=msg
)
