#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('IPySPSolutionSaverExtension',
           'IPySPSolutionLoaderExtension')

from pyomo.common.plugin import Interface

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
