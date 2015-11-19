#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import os
import copy
import six

import pyutilib.services
import pyutilib.common
import pyutilib.misc

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import TransformationFactory
from pyomo.solvers.mockmip import MockMIP

import logging
logger = logging.getLogger('pyomo.solvers')


class ASL(SystemCallSolver):
    """A generic optimizer that uses the AMPL Solver Library to interface with applications.
    """

    pyomo.util.plugin.alias('asl', doc='Interface for solvers using the AMPL Solver Library')

    def __init__(self, **kwds):
        #
        # Call base constructor
        #
        if not 'type' in kwds:
            kwds["type"] = "asl"
        SystemCallSolver.__init__(self, **kwds)
        self._metasolver = True
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

    def _default_results_format(self, prob_format):
        return ResultsFormat.sol

    def _default_executable(self):
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

    def _get_version(self):
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
        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.\
                             create_tempfile(suffix="_%s.log" % self.options.solver)

        #
        # Define solution file
        #
        if self._soln_file is not None:
            # the solution file can not be redefined
            logger.warning("The 'soln_file' keyword will be ignored "
                           "for solver="+self.type)
        fname = problem_files[0]
        if '.' in fname:
            tmp = fname.split('.')
            fname = '.'.join(tmp[:-1])
        self._soln_file = fname+".sol"

        #
        # Define results file (since an external parser is used)
        #
        self._results_file = self._soln_file

        #
        # Define command line
        #
        env=copy.copy(os.environ)
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

        cmd = [executable, '-s', problem_files[0]]
        if self._timer:
            cmd.insert(0, self._timer)
        #
        # GAH: I am going to re-add the code by Zev that passed options through
        # to the command line. Setting the environment variable in this way does
        # NOT work for solvers like cplex and gurobi because the are looking for
        # an environment variable called cplex_options / gurobi_options. However
        # the options.solver name for these solvers is cplexamp / gurobi_ampl
        # (which creates a cplexamp_options and gurobi_ampl_options env variable).
        # Because of this, I think the only reliable way to pass options for any
        # solver is by using the command line
        #
        opt=[]
        for key in self.options:
            if key == 'solver':
                continue
            if isinstance(self.options[key],six.string_types) and \
               (' ' in self.options[key]):
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

        return pyutilib.misc.Bunch(cmd=cmd, log_file=self._log_file, env=env)

    def _presolve(self, *args, **kwds):
        if not isinstance(args[0], six.string_types):
            self._instance = args[0]
            xfrm = TransformationFactory('mpec.nl')
            xfrm.apply_to(self._instance)
            if len(self._instance._transformation_data['mpec.nl'].compl_cuids) == 0:
                # There were no complementarity conditions
                # so we don't hold onto the instance
                self._instance = None
            else:
                args = (self._instance,)
        else:
            self._instance = None
        #
        SystemCallSolver._presolve(self, *args, **kwds)

    def _postsolve(self):
        #
        # Reclassify complementarity components
        #
        mpec=False
        if not self._instance is None:
            from pyomo.mpec import Complementarity
            for cuid in self._instance._transformation_data['mpec.nl'].compl_cuids:
                mpec=True
                cobj = cuid.find_component(self._instance)
                cobj.parent_block().reclassify_component_type(cobj, Complementarity)
        #
        self._instance = None
        return SystemCallSolver._postsolve(self)


class MockASL(ASL,MockMIP):
    """A Mock ASL solver used for testing
    """

    pyomo.util.plugin.alias('_mock_asl')

    def __init__(self, **kwds):
        try:
            ASL.__init__(self,**kwds)
        except pyutilib.common.ApplicationError: #pragma:nocover
            pass                        #pragma:nocover
        MockMIP.__init__(self,"asl")

    def available(self, exception_flag=True):
        return ASL.available(self,exception_flag)

    def create_command_line(self,executable, problem_files):
        command = ASL.create_command_line(self,
                                          executable,
                                          problem_files)
        MockMIP.create_command_line(self,
                                    executable,
                                    problem_files)
        return command

    def executable(self):
        return MockMIP.executable(self)

    def _execute_command(self,cmd):
        return MockMIP._execute_command(self,cmd)
