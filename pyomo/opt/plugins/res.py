#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Class for reading results with the SolverResults object
#


from pyomo.opt.base import results
from pyomo.opt.base.formats import ResultsFormat
from pyomo.opt import SolverResults


@results.ReaderFactory.register(str(ResultsFormat.yaml))
class ResultsReader_yaml(results.AbstractResultsReader):
    """
    Class that reads in a *.yml file and generates a
    SolverResults object.
    """

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



@results.ReaderFactory.register(str(ResultsFormat.json))
class ResultsReader_json(results.AbstractResultsReader):
    """
    Class that reads in a *.jsn file and generates a
    SolverResults object.
    """

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

