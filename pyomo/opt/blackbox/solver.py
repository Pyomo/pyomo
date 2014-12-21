#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.opt.base import *


class COLINSolver(OptSolver):
    """An optimizer that can optimize the pyomo.opt.colin.problem.OptProblem object"""

    def __init__(self, **kwds):
        """ Constructor """
        OptSolver.__init__(self,**kwds)
        self._valid_problem_formats=[ProblemFormat.colin_optproblem]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.colin_optproblem] = [ResultsFormat.osrl,ResultsFormat.results]
