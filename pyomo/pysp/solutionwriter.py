#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['ISolutionWriterExtension']

from pyomo.util.plugin import Interface


class ISolutionWriterExtension(Interface):

    def write(self, scenario_tree, output_file_prefix):
        """Called with a ScenarioTree type object."""
        pass
