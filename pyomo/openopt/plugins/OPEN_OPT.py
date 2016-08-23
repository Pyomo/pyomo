#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import logging

from pyutilib.common import ApplicationError
import pyutilib.misc
import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt import (SolverResults,
                       TerminationCondition,
                       SolutionStatus,
                       Solution,
                       ProblemSense)
import pyomo.openopt.func_designer

import six
from six import iteritems

try:
    import FuncDesigner
    FD_available=True
except:
    FD_available=False
try:
    import openopt
    OO_available=True
except:
    OO_available=False

logger = logging.getLogger('pyomo.core')

class OpenOptSolver(OptSolver):
    """A generic interface to OpenOpt solvers"""

    pyomo.util.plugin.alias('openopt', doc='Interface to OpenOpt solvers')

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "OpenOpt"
        OptSolver.__init__(self, **kwds)
        self._metasolver = True
        #
        # Setup valid problem formats, and valid results for each problem format
        #
        self._problem_format = ProblemFormat.FuncDesigner
        self._valid_problem_formats=[]
        self._valid_result_formats = {}

        # Note: Undefined capabilities default to 'None'
        self._capabilities = pyutilib.misc.Options()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False
        self._capabilities.sosn = False

    def available(self, exception_flag=True):
        if not (FD_available and OO_available):
            if exception_flag:
                raise ApplicationError('Cannot execute solver without FuncDesigner and OpenOpt installed')
            return False
        return OptSolver.available(self, exception_flag)

    def _convert_problem(self, args, pformat, valid_pformats, **kwds):

        if len(kwds):
            logger.warning(
                "The following keywords will be ignored by solver="+self.type+":\n\t"+
                "\n\t".join("%s = %s" % (k,v) for k,v in iteritems(kwds)))

        if self._problem is not None:
            return (self.problem,ProblemFormat.colin_optproblem,None)
        self._instance = args[0]
        self._problem = pyomo.openopt.func_designer.Pyomo2FuncDesigner(args[0])
        return (self._problem, ProblemFormat.FuncDesigner, None)

    def _apply_solver(self):
        try:
            self._ans = self._problem.minimize(self._problem.f,
                                              self._problem.initial_point,
                                              solver=self.options.solver)
        except openopt.OpenOptException:
            e = sys.exc_info()[1]
            raise RuntimeError(str(e))
        self._symbol_map = self._problem._symbol_map
        return pyutilib.misc.Bunch(rc=None, log=None)

    def _postsolve(self):
        results = SolverResults()

        #print 'ANS', dir(self._ans), 
        #print self._ans.evals
        #print self._ans.ff
        #print self._ans.rf
        #print self._ans.xf

        solv = results.solver
        solv.name = self.options.subsolver
        #solv.status = self._glpk_get_solver_status()
        #solv.memory_used = "%d bytes, (%d KiB)" % (peak_mem, peak_mem/1024)
        solv.wallclock_time = self._ans.elapsed['solver_time']
        solv.cpu_time = self._ans.elapsed['solver_cputime']

        solv.termination_message = self._ans.msg
        istop = self._ans.istop
        if istop == openopt.kernel.setDefaultIterFuncs.SMALL_DF:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.locallyOptimal

        elif istop == openopt.kernel.setDefaultIterFuncs.SMALL_DELTA_X:
            solv.termination_condition = TerminationCondition.minStepLength
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.SMALL_DELTA_F:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.FVAL_IS_ENOUGH:
            solv.termination_condition = TerminationCondition.minFunctionValue
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.MAX_NON_SUCCESS:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.unsure

        elif istop == openopt.kernel.setDefaultIterFuncs.USER_DEMAND_STOP:
            solv.termination_condition = TerminationCondition.userInterrupt
            sstatus = SolutionStatus.bestSoFar

        elif istop == openopt.kernel.setDefaultIterFuncs.BUTTON_ENOUGH_HAS_BEEN_PRESSED:
            solv.termination_condition = TerminationCondition.userInterrupt
            sstatus = SolutionStatus.bestSoFar

        elif istop == openopt.kernel.setDefaultIterFuncs.SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.unsure

        elif istop == openopt.kernel.setDefaultIterFuncs.UNDEFINED:
            solv.termination_condition = TerminationCondition.unknown
            sstatus = SolutionStatus.unsure

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_NAN_IN_X:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.unknown

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_LINE_SEARCH_FAILED:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.error

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_MAX_ITER_REACHED:
            solv.termination_condition = TerminationCondition.maxIterations
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_MAX_CPU_TIME_REACHED:
            solv.termination_condition = TerminationCondition.maxTimeLimit
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_MAX_TIME_REACHED:
            solv.termination_condition = TerminationCondition.maxTimeLimit
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_MAX_FUN_EVALS_REACHED:
            solv.termination_condition = TerminationCondition.maxEvaluations
            sstatus = SolutionStatus.stoppedByLimit

        elif istop == openopt.kernel.setDefaultIterFuncs.IS_ALL_VARS_FIXED:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.unknown

        elif istop == openopt.kernel.setDefaultIterFuncs.FAILED_TO_OBTAIN_MOVE_DIRECTION:
            solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.error

        elif istop == openopt.kernel.setDefaultIterFuncs.USER_DEMAND_EXIT:
            solv.termination_condition = TerminationCondition.userInterrupt
            sstatus = SolutionStatus.bestSoFar

        elif istop == -100:
            #solv.termination_condition = TerminationCondition.other
            sstatus = SolutionStatus.error

        else:
            raise ApplicationError("Unexpected OpenOpt termination code: '%d'" % istop)

        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables = self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables = self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives

        from pyomo.core import maximize
        if self._problem.sense == maximize:
            prob.sense = ProblemSense.maximize
        else:
            prob.sense = ProblemSense.minimize

        if not sstatus in ( SolutionStatus.error, ):
            soln = Solution()
            soln.status = sstatus

            if type(self._ans.ff) in (list, tuple):
                oval = float(self._ans.ff[0])
            else:
                oval = float(self._ans.ff)
            if self._problem.sense == maximize:
                soln.objective[self._problem._f_name[0]] = {'Value': - oval}
            else: 
                soln.objective[self._problem._f_name[0]] = {'Value': oval}

            for var_label in self._ans.xf.keys():
                if self._ans.xf[var_label].is_integer():
                    soln.variable[ var_label.name ] = {'Value': int(self._ans.xf[var_label])}
                else:
                    soln.variable[ var_label.name ] = {'Value': float(self._ans.xf[var_label])}

            results.solution.insert( soln )

        self._instance.solutions.add_symbol_map(self._symbol_map)
        self._smap_id = id(self._symbol_map)

        self._instance = None
        self._symbol_map = None
        self._problem = None
        return results

