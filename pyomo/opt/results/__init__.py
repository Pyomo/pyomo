#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#from old_results import *
from pyomo.opt.results.container import *
import pyomo.opt.results.problem
from pyomo.opt.results.solver import SolverStatus, TerminationCondition
from pyomo.opt.results.problem import ProblemSense
from pyomo.opt.results.solution import SolutionStatus, Solution
from pyomo.opt.results.results_ import SolverResults
