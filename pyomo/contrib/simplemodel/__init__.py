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

from pyomo.common.deprecation import deprecation_warning, in_testing_environment

try:
    deprecation_warning(
        "The use of pyomo.contrib.simple model is deprecated. "
        "This capability is now supported in the pyomo_simplemodel "
        "package, which is included in the pyomo_community distribution.",
        version='5.6.9',
    )
    from pyomocontrib_simplemodel import *
except ImportError:
    # Only raise the exception if nose/pytest/sphinx are NOT running
    # (otherwise test discovery can result in exceptions)
    if not in_testing_environment():
        raise RuntimeError("The pyomocontrib_simplemodel package is not installed.")
