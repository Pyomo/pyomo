#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.download import DownloadFactory
from pyomo.common.extensions import ExtensionBuilderFactory
from .getMCPP import get_mcpp
from .build import MCPPBuilder

def load():
    DownloadFactory.register('mcpp')(get_mcpp)
    ExtensionBuilderFactory.register('mcpp')(MCPPBuilder)

