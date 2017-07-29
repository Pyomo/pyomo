#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Define plugins used for defining OptProblem IO formats.
"""

from pyomo.util.plugin import *

class IBlackBoxOptProblemIO(Interface):
    """An interface that defines reader/writer for a particular IO format"""

    def read(self, filename, point):
        """
        Read a point and request information.
        This method returns a tuple: point, requests
        """

    def write(self, filename, response):
        """
        Write response information to a file.
        """
