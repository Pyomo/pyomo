#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('IPySPSolutionSaverExtension',
           'IPySPSolutionLoaderExtension')

from pyomo.util.plugin import Interface

class IPySPSolutionSaverExtension(Interface):

    def save(self, manager):
        """Save solution data from a scenario tree manager.
        Returns True only if data was saved."""
        return False

class IPySPSolutionLoaderExtension(Interface):

    def load(self, manager):
        """Load solution data into a scenario tree manager.
        Returns True only if data was loaded."""
        return True
