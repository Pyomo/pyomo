#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = [ 'IProblemWriter', 'AbstractProblemWriter', 'WriterFactory' ]

from pyomo.util.plugin import *


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
