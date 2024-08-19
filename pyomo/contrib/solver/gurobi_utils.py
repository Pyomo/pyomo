#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import ConfigValue


class GurobiConfigMixin:
    """
    Mixin class for Gurobi-specific configurations
    """

    def __init__(self):
        self.use_mipstart: bool = self.declare(
            'use_mipstart',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Gurobi.",
            ),
        )
