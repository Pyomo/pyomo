#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

#
# Class for reading results with the SolverResults object
#

import os
import sys
import re

from pyomo.opt.base import results
from pyomo.opt.base.formats import *
from pyomo.opt import SolverResults
from pyomo.misc.plugin import alias


class ResultsReader_yaml(results.AbstractResultsReader):
    """
    Class that reads in a *.yml file and generates a
    SolverResults object.
    """

    alias(str(ResultsFormat.yaml))

    def __init__(self):
        results.AbstractResultsReader.__init__(self, ResultsFormat.yaml)

    def __call__(self, filename, res=None, soln=None, suffixes=[]):
        """
        Parse a *.results file
        """
        if res is None:
            res = SolverResults()
        #
        res.read(filename, using_yaml=True)
        return res



class ResultsReader_json(results.AbstractResultsReader):
    """
    Class that reads in a *.jsn file and generates a
    SolverResults object.
    """

    alias(str(ResultsFormat.json))

    def __init__(self):
        results.AbstractResultsReader.__init__(self, ResultsFormat.json)

    def __call__(self, filename, res=None, soln=None, suffixes=[]):
        """
        Parse a *.results file
        """
        if res is None:
            res = SolverResults()
        #
        res.read(filename, using_yaml=False)
        return res

