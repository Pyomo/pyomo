#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#from old_results import *
from pyomo.opt.results.container import *
import pyomo.opt.results.problem
from pyomo.opt.results.solver import SolverStatus, TerminationCondition
from pyomo.opt.results.problem import ProblemSense
from pyomo.opt.results.solution import SolutionStatus, Solution
from pyomo.opt.results.results_ import SolverResults
