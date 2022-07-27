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

from .utils.ipopt_solver_wrapper import *

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.contrib.parmest.ipopt_solver_wrapper module has been moved to '
    'pyomo.contrib.parmest.utils.ipopt_solver_wrapper. Please update your import',
    version='TBD')
