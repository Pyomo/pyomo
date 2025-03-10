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
from pyomo.common.deprecation import moved_module

moved_module(
    'pyomo.repn.tests.ampl.nl_diff',
    'pyomo.repn.tests.nl_diff',
    version='6.6.0',
    remove_in='6.6.1',
)
del moved_module
