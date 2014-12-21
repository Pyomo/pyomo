#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

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
