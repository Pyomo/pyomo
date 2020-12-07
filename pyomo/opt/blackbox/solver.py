#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.opt.base import *


class COLINSolver(OptSolver):
    """An optimizer that can optimize the pyomo.opt.colin.problem.OptProblem object"""

    def __init__(self, **kwds):
        """ Constructor """
        OptSolver.__init__(self,**kwds)
        self._assert_available=True
        self._valid_problem_formats=[ProblemFormat.colin_optproblem]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.colin_optproblem] = [ResultsFormat.osrl,ResultsFormat.results]
