#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


import re
import os
import copy

import pyutilib.services
import pyutilib.common
import pyutilib.common
import pyutilib.misc

import pyomo.misc.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.solvers.mockmip import MockMIP

import logging
logger = logging.getLogger('pyomo.solvers')

try:
    unicode
except:
    basestring = str


class ASL(SystemCallSolver):
    """A generic optimizer that uses the AMPL Solver Library to interface with applications.
    """

    pyomo.misc.plugin.alias('asl', doc='Interface for solvers using the AMPL Solver Library')

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        kwds["type"] = "asl"
        SystemCallSolver.__init__(self, **kwds)
        #
        # Setup valid problem formats, and valid results for each problem format.
        # Also set the default problem and results formats.
        #
        self._valid_problem_formats=[ProblemFormat.nl]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.nl] = [ResultsFormat.sol]
        self.set_problem_format(ProblemFormat.nl)
        #
        # Note: Undefined capabilities default to 'None'
        #
        self._capabilities = pyutilib.misc.Options()
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True
        #
        # This interface is always availble
        #
        ##self._assert_available = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.sol

    def executable(self):
        #
        # We register the ASL executables dynamically, since _any_ ASL solver could be
        # executed by this solver.
        #
        if self.options.solver is None:
            logger.warning("No solver option specified for ASL solver interface")
            return None
        try:
            pyutilib.services.register_executable(self.options.solver)
        except:
            logger.warning("No solver option specified for ASL solver interface")
            return None
        executable = pyutilib.services.registered_executable(self.options.solver)
        if executable is None:
            logger.warning("Could not locate the '%s' executable, which is required for solver %s" % (self.options.solver, self.name))
            self.enable = False
            return None
        return executable.get_path()

    def version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        results = pyutilib.subprocess.run( [solver_exec,"-v"], timelimit=1 )
        return _extract_version(results[1])

    def create_command_line(self, executable, problem_files):
 
        assert(self._problem_format == ProblemFormat.nl)
        assert(self._results_format == ResultsFormat.sol)

        #
        # Define log file
        #
        if self.log_file is None:
            self.log_file = pyutilib.services.TempfileManager.create_tempfile(suffix="_asl.log")
        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            if len(tmp) > 2:
                fname = '.'.join(tmp[:-1])
            else:
                fname = tmp[0]
        self.soln_file = fname+".sol"

        #
        # Define results file
        #
        self.results_file = self.soln_file
        
        #
        # Define command line
        #
        env=copy.copy(os.environ)

        #
        # Merge the COOPR_AMPLFUNC (externals defined within
        # Pyomo/Pyomo) with any user-specified external function
        # libraries
        #
        if 'COOPR_AMPLFUNC' in env:
            if 'AMPLFUNC' in env:
                env['AMPLFUNC'] += "\n" + env['COOPR_AMPLFUNC']
            else:
                env['AMPLFUNC'] = env['COOPR_AMPLFUNC']

        cmd = [executable, '-s', problem_files[0]]
        if self._timer:
            cmd.insert(0, self._timer)
        
        # GAH: I am going to re-add the code by Zev that passed options through
        # to the command line. Setting the environment variable in this way does
        # NOT work for solvers like cplex and gurobi because the are looking for 
        # an environment variable called cplex_options / gurobi_options. However
        # the options.solver name for these solvers is cplexamp / gurobi_ampl 
        # (which creates a cplexamp_options and gurobi_ampl_options env variable).
        # Because of this, I think the only reliable way to pass options for any 
        # solver is by using the command line
        opt=[]
        for key in self.options:
            if key is 'solver':
                continue
            if isinstance(self.options[key],basestring) and ' ' in self.options[key]:
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
            
        return pyutilib.misc.Bunch(cmd=cmd, log_file=self.log_file, env=env)

    def Xprocess_soln_file(self,results):
        """
        Process the SOL file
        """
        if os.path.exists(self.soln_file):
            results_reader = ReaderFactory(ResultsFormat.sol)
            results = results_reader(self.soln_file, results, results.solution(0))
            return


class MockASL(ASL,MockMIP):
    """A Mock ASL solver used for testing
    """

    pyomo.misc.plugin.alias('_mock_asl')

    def __init__(self, **kwds):
        try:
            ASL.__init__(self,**kwds)
        except pyutilib.common.ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"asl")

    def available(self, exception_flag=True):
        return ASL.available(self,exception_flag)

    def create_command_line(self,executable,problem_files):
        command = ASL.create_command_line(self,executable,problem_files)
        MockMIP.create_command_line(self,executable,problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)
