#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
