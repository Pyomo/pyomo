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
"""Configuration options for incidence graph generation
"""

import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum


class IncidenceMethod(enum.Enum):
    """Methods for identifying variables that participate in expressions"""

    identify_variables = 0
    """Use ``pyomo.core.expr.visitor.identify_variables``"""

    standard_repn = 1
    """Use ``pyomo.repn.standard_repn.generate_standard_repn``"""


_include_fixed = ConfigValue(
    default=False,
    domain=bool,
    description="Include fixed variables",
    doc=(
        "Flag indicating whether fixed variables should be included in the"
        " incidence graph"
    ),
)


_linear_only = ConfigValue(
    default=False,
    domain=bool,
    description="Identify variables that participate linearly",
    doc=(
        "Flag indicating whether only variables that participate linearly should"
        " be included. Note that these are included even if they participate"
        " nonlinearly as well."
    ),
)


_method = ConfigValue(
    default=IncidenceMethod.standard_repn,
    domain=InEnum(IncidenceMethod),
    description="Method used to identify incident variables",
)


IncidenceConfig = ConfigDict()
"""Options for incidence graph generation

- ``include_fixed`` -- Flag indicating whether fixed variables should be included
  in the incidence graph
- ``linear_only`` -- Flag indicating whether only variables that participate linearly
  should be included. Note that these are included even if they participate
  nonlinearly as well
- ``method`` -- Method used to identify incident variables. Must be a value of the
  ``IncidenceMethod`` enum.

"""


IncidenceConfig.declare("include_fixed", _include_fixed)


IncidenceConfig.declare("linear_only", _linear_only)


IncidenceConfig.declare("method", _method)
