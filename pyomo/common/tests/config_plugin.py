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
from pyomo.common.config import ConfigDict, ConfigValue


def get_configuration(config):
    ans = ConfigDict()
    ans.declare('key1', ConfigValue(default=0, domain=int))
    ans.declare('key2', ConfigValue(default=5, domain=str))
    return ans(config)
