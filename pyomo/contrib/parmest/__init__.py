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

#
# declare deprecation paths for removed modules
#
from pyomo.common.deprecation import relocated_module_attribute, moved_module

moved_module(
    'pyomo.contrib.parmest.ipopt_solver_wrapper',
    'pyomo.contrib.parmest.utils.ipopt_solver_wrapper',
    version='6.4.2',
)
relocated_module_attribute(
    'create_ef', 'pyomo.contrib.parmest.utils.create_ef', version='6.4.2'
)
relocated_module_attribute(
    'ipopt_solver_wrapper',
    'pyomo.contrib.parmest.utils.ipopt_solver_wrapper',
    version='6.4.2',
)
relocated_module_attribute(
    'mpi_utils', 'pyomo.contrib.parmest.utils.mpi_utils', version='6.4.2'
)
relocated_module_attribute(
    'scenario_tree', 'pyomo.contrib.parmest.utils.scenario_tree', version='6.4.2'
)
del relocated_module_attribute, moved_module
