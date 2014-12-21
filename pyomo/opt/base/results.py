#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = [ 'IResultsReader', 'AbstractResultsReader', 'ReaderFactory' ]

from pyomo.util.plugin import *


class IResultsReader(Interface):
    """Interface for classes that can read optimization results."""

ReaderFactory = CreatePluginFactory(IResultsReader)

class AbstractResultsReader(Plugin):
    """Base class that can read optimization results."""

    implements(IResultsReader)

    def __init__(self, results_format):
        Plugin.__init__(self)
        self.format=results_format

    def __call__(self, filename, res=None, suffixes=[]): #pragma:nocover
        raise TypeError("Method __call__ undefined in reader for format "+str(self.format))
