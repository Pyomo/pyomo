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
# declare deprecation paths for removed modules and attributes
#
from pyomo.common.deprecation import moved_module

moved_module(
    "pyomo.contrib.simplemodel",
    "pyomocontrib_simplemodel",
    msg="The use of pyomo.contrib.simplemodel is deprecated. "
    "This capability is now supported in the pyomocontrib_simplemodel "
    "package, which is included in the pyomo_community distribution.",
    version='5.6.9',
)
del moved_module
