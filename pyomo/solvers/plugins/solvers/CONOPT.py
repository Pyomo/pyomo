#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

from pyomo.common import Executable
from pyomo.common.collections import Options, Bunch
from pyomo.common.tempfiles import TempfileManager
from pyutilib.subprocess import run

from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import SolverStatus
from pyomo.opt.solver import  SystemCallSolver

import logging
logger = logging.getLogger('pyomo.solvers')

try:
    unicode
except:
    basestring = str


@SolverFactory.register('conopt', doc='The CONOPT NLP solver')
class CONOPT(SystemCallSolver):
    """
    An interface to the CONOPT optimizer that uses the AMPL Solver Library.
    """


    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "conopt"
        super(CONOPT, self).__init__(**kwds)
        #
        # Setup valid problem formats, and valid results for each problem format
        # Also set the default problem and results formats.
        #
        self._valid_problem_formats=[ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self.set_problem_format(ProblemFormat.nl)

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.sol

    def _default_executable(self):
        executable = Executable("conopt")
        if not executable:
            logger.warning("Could not locate the 'conopt' executable, "
                           "which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        results = run( [solver_exec], timelimit=1 )
        return _extract_version(results[1])

    def create_command_line(self, executable, problem_files):

        assert(self._problem_format == ProblemFormat.nl)
        assert(self._results_format == ResultsFormat.sol)

        #
        # Define log file
        #
        if self._log_file is None:
            self._log_file = TempfileManager.\
                             create_tempfile(suffix="_conopt.log")

        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            if len(tmp) > 2:
                fname = '.'.join(tmp[:-1])
            else:
                fname = tmp[0]
        self._soln_file = fname+".sol"

        #
        # Define results file (since an external parser is used)
        #
        self._results_file = self._soln_file

        #
        # Define command line
        #
        env=os.environ.copy()
        #
        # Merge the PYOMO_AMPLFUNC (externals defined within
        # Pyomo/Pyomo) with any user-specified external function
        # libraries
        #
        if 'PYOMO_AMPLFUNC' in env:
            if 'AMPLFUNC' in env:
                env['AMPLFUNC'] += "\n" + env['PYOMO_AMPLFUNC']
            else:
                env['AMPLFUNC'] = env['PYOMO_AMPLFUNC']

        cmd = [executable, problem_files[0], '-AMPL']
        if self._timer:
            cmd.insert(0, self._timer)

        # GAH: I am going to re-add the code by Zev that passed options through
        # to the command line. I'm not sure what solvers this method of passing options
        # through the envstr variable works for, but it does not seem to work for cplex
        # or gurobi
        opt=[]
        for key in self.options:
            if key == 'solver':
                continue
            if isinstance(self.options[key], basestring) and ' ' in self.options[key]:
                opt.append(key+"=\""+str(self.options[key])+"\"")
                cmd.append(str(key)+"="+str(self.options[key]))
            elif key == 'subsolver':
                opt.append("solver="+str(self.options[key]))
                cmd.append(str(key)+"="+str(self.options[key]))
            else:
                opt.append(key+"="+str(self.options[key]))
                cmd.append(str(key)+"="+str(self.options[key]))

        envstr = "%s_options" % self.options.solver
        # Merge with any options coming in through the environment
        env[envstr] = " ".join(opt)

        return Bunch(cmd=cmd, log_file=self._log_file, env=env)

    def _postsolve(self):
        results = super(CONOPT, self)._postsolve()
        # Hack so that the locally optimal termination
        # condition for CONOPT does not trigger a warning.
        # For some reason it sets the solver_results_num to
        # 100 in this case, which is reserved for cases
        # where "optimal solution indicated, but error likely".
        if results.solver.id == 100 and \
            'Locally optimal' in results.solver.message:
            results.solver.status = SolverStatus.ok
        return results
