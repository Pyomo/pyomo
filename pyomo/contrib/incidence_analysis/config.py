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
from pyomo.common.config import ConfigBlock, ConfigValue, InEnum


class IncidenceMethod(enum.Enum):
    identify_variables = 0
    standard_repn = 1


IncidenceConfig = ConfigBlock()


IncidenceConfig.declare(
    "include_fixed",
    ConfigValue(default=False, domain=bool, description="Include fixed variables"),
)


IncidenceConfig.declare(
    "linear_only",
    ConfigValue(
        default=False,
        domain=bool,
        description="Only identify variables that participate linearly",
    ),
)


IncidenceConfig.declare(
    "method",
    ConfigValue(
        default=IncidenceMethod.standard_repn,
        domain=InEnum(IncidenceMethod),
        description="Method used to identify incident variables",
    ),
)
