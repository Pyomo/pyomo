#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [ 'IProblemWriter', 'AbstractProblemWriter', 'WriterFactory', 'ProblemConfigFactory', 'BaseProblemConfig' ]

from pyomo.util.plugin import *


class IProblemConfig(Interface):
    """Interface for classes that create configuration blocks."""

ProblemConfigFactory = CreatePluginFactory(IProblemConfig)


class BaseProblemConfig(Plugin):
    """Base class for plugins generating problem configurations"""

    implements(IProblemConfig)

    def config_block(self):
        pass


class IProblemWriter(Interface):
    """Interface for classes that can write optimization problems."""

WriterFactory = CreatePluginFactory(IProblemWriter)


class AbstractProblemWriter(Plugin):
    """Base class that can write optimization problems."""

    implements(IProblemWriter)

    def __init__(self, problem_format): #pragma:nocover
        Plugin.__init__(self)
        self.format=problem_format

    def __call__(self, model, filename, solver_capability, **kwds): #pragma:nocover
        raise TypeError("Method __call__ undefined in writer for format "+str(self.format))
