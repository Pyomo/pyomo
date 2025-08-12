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

from pyomo.repn.standard_repn import StandardRepn, generate_standard_repn
from pyomo.repn.standard_aux import compute_standard_repn

from pyomo.common.deprecation import moved_module

moved_module(
    "pyomo.repn.parameterized_linear",
    "pyomo.repn.parameterized",
    msg="The pyomo.repn.parameterized_linear module is deprecated.  "
    "Import the ParameterizedLinearRepnVisitor from pyomo.repn.parameterized",
    version='6.9.3',
)
moved_module(
    "pyomo.repn.parameterized_quadratic",
    "pyomo.repn.parameterized",
    msg="The pyomo.repn.parameterized_quadratic module is deprecated.  "
    "Import the ParameterizedQuadraticRepnVisitor from pyomo.repn.parameterized",
    version='6.9.3',
)

del moved_module
