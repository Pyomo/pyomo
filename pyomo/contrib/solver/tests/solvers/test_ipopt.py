# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import datetime
import os
import stat
import subprocess
import sys
import time
import threading
from contextlib import contextmanager

import pyomo.environ as pyo
from pyomo.common.envvar import is_windows
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict, ADVANCED_OPTION
from pyomo.common.errors import ApplicationError, MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.timing import HierarchicalTimer
import pyomo.contrib.solver.solvers.ipopt as ipopt
from pyomo.contrib.solver.common.util import NoSolutionError, NoOptimalSolutionError
from pyomo.contrib.solver.common.results import TerminationCondition, SolutionStatus
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.common import unittest, Executable
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.repn.util import FileDeterminism

ipopt_available = ipopt.Ipopt().available()


@contextmanager
def windows_tee_buffer(size=1 << 20):
    """Temporarily increase TeeStream buffer size on Windows"""
    if not sys.platform.startswith("win"):
        # Only windows has an issue
        yield
        return
    import pyomo.common.tee as tee

    old = tee._pipe_buffersize
    tee._pipe_buffersize = size
    try:
        yield
    finally:
        tee._pipe_buffersize = old


@unittest.pytest.mark.solver("ipopt")
class TestIpoptSolverConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = ipopt.IpoptConfig()
        # Should be inherited
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)
        # Unique to this object
        self.assertIsInstance(config.executable, type(Executable('path')))
        self.assertIsInstance(config.writer_config, type(NLWriter.CONFIG()))

    def test_custom_instantiation(self):
        config = ipopt.IpoptConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertIsNone(config.time_limit)
        # Default should be `ipopt`
        self.assertEqual('ipopt', config.executable._registered_name)
        if ipopt_available:
            self.assertIsNotNone(config.executable.path())
            self.assertIn('ipopt', str(config.executable))
        # Set to a totally bogus path
        config.executable = Executable('/bogus/path')
        self.assertIsNone(config.executable.executable)
        self.assertFalse(config.executable.available())


@unittest.pytest.mark.solver("ipopt")
class TestIpoptSolutionLoader(unittest.TestCase):
    def test_get_reduced_costs_error(self):
        loader = ipopt.IpoptSolutionLoader(
            ipopt.ASLSolFileData(), NLWriterInfo(eliminated_vars=[1])
        )
        with self.assertRaisesRegex(
            MouseTrap, "Complete reduced costs are not available"
        ):
            loader.get_reduced_costs()

    def test_get_duals_error(self):
        loader = ipopt.IpoptSolutionLoader(
            ipopt.ASLSolFileData(), NLWriterInfo(eliminated_vars=[1])
        )
        with self.assertRaisesRegex(MouseTrap, "Complete duals are not available"):
            loader.get_duals()


@unittest.pytest.mark.solver("ipopt")
class TestIpoptInterface(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_command_line_options(self):
        result = subprocess.run(
            [str(ipopt.Ipopt.CONFIG.executable), '-='],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        options = []
        for line in output.splitlines():
            option_name = line.split()[0] if line.strip() else ''
            if option_name:
                options.append(option_name)
        self.assertEqual(sorted(ipopt.ipopt_command_line_options), sorted(options))

    def test_class_member_list(self):
        opt = ipopt.Ipopt()
        expected_list = [
            'CONFIG',
            'config',
            'api_version',
            'available',
            'has_linear_solver',
            'is_persistent',
            'solve',
            'version',
            'name',
        ]
        method_list = [method for method in dir(opt) if not method.startswith('_')]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = ipopt.Ipopt()
        self.assertFalse(opt.is_persistent())
        self.assertEqual(opt.name, 'ipopt')
        self.assertEqual(opt.CONFIG, opt.config)
        if ipopt_available:
            self.assertIsNotNone(opt.version())
            self.assertTrue(opt.available())
        else:
            self.assertIsNone(opt.version())
            self.assertFalse(opt.available())

    def test_context_manager(self):
        with ipopt.Ipopt() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertEqual(opt.name, 'ipopt')
            self.assertEqual(opt.CONFIG, opt.config)
            if ipopt_available:
                self.assertIsNotNone(opt.version())
                self.assertTrue(opt.available())
            else:
                self.assertIsNone(opt.version())
                self.assertFalse(opt.available())

    def test_get_version(self):
        if ipopt_available:
            ver = ipopt.Ipopt().version()
            self.assertIsInstance(ver, tuple)
            self.assertEqual(len(ver), 3)
            self.assertTrue(all(isinstance(_, int) for _ in ver))

        _cache = ipopt.Ipopt._exe_cache
        try:
            with TempfileManager.new_context() as TMP:
                dname = TMP.mkdtemp()

                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test1')
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({None: None}, ipopt.Ipopt._exe_cache)

                # the rest of this test is designed to work on *nix:
                if sys.platform.startswith("win"):
                    return

                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test2')
                with open(fname, 'w') as F:
                    F.write(f"#!{sys.executable}\nimport sys\nsys.exit(0)\n")
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({None: None}, ipopt.Ipopt._exe_cache)

                # Found an executable, but --version errors
                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test3')
                with open(fname, 'w') as F:
                    F.write(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n")
                os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({fname: None}, ipopt.Ipopt._exe_cache)

                # Found an executable, but --version doesn't return anything
                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test4')
                with open(fname, 'w') as F:
                    F.write(f"#!{sys.executable}\nimport sys\nsys.exit(0)\n")
                os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({fname: None}, ipopt.Ipopt._exe_cache)

                # Missing "ipopt"
                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test5')
                with open(fname, 'w') as F:
                    F.write(
                        f"#!{sys.executable}\nprint('cbc 1.2.3 ASL')\n"
                        "import sys\nsys.exit(0)\n"
                    )
                os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({fname: None}, ipopt.Ipopt._exe_cache)

                # The version doesn't parse correctly
                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test6')
                with open(fname, 'w') as F:
                    F.write(
                        f"#!{sys.executable}\nprint('Ipopt 1.2.3a ASL')\n"
                        "import sys\nsys.exit(0)\n"
                    )
                os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.NotFound, solver.available())
                self.assertIsNone(solver.version())
                self.assertEqual({fname: None}, ipopt.Ipopt._exe_cache)

                # This looks like an Ipopt solver...
                ipopt.Ipopt._exe_cache = {}
                fname = os.path.join(dname, 'test7')
                with open(fname, 'w') as F:
                    F.write(
                        f"#!{sys.executable}\nprint('Ipopt 1.2.3 ASL')\n"
                        "import sys\nsys.exit(0)\n"
                    )
                os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver = ipopt.Ipopt(executable=fname)
                self.assertEqual({}, ipopt.Ipopt._exe_cache)
                self.assertEqual(ipopt.Availability.FullLicense, solver.available())
                self.assertEqual(solver.version(), (1, 2, 3))
                self.assertEqual({fname: (1, 2, 3)}, ipopt.Ipopt._exe_cache)

        finally:
            ipopt.Ipopt._exe_cache = _cache

    def test_parse_output(self):
        # Old ipopt style (<=3.13)
        # Note: we are removing the URLs from the baseline because they
        # do not impact the test (and checking the URLs is fragile)
        output = """Ipopt 3.13.2:

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit

This version of Ipopt was compiled from source code available at
    https://github.com/IDAES/Ipopt as part of the Institute for the Design of
    Advanced Energy Systems Process Systems Engineering Framework (IDAES PSE
    Framework) Copyright (c) 2018-2019. See https://github.com/IDAES/idaes-pse.

This version of Ipopt was compiled using HSL, a collection of Fortran codes
    for large-scale scientific computation.  All technical papers, sales and
    publicity material resulting from use of the HSL codes within IPOPT must
    contain the following acknowledgement:
        HSL, a collection of Fortran codes for large-scale scientific
        computation. See
******************************************************************************

This is Ipopt version 3.13.2, running with linear solver ma27.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:        3

Total number of variables............................:        2
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  5.6500000e+01 0.00e+00 1.00e+02  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4669972e-01 0.00e+00 2.22e-01  -1.0 7.40e-01    -  1.00e+00 1.00e+00f  1
   2  1.6256267e-01 0.00e+00 2.04e+00  -1.7 1.48e+00    -  1.00e+00 2.50e-01f  3
   3  8.6119444e-02 0.00e+00 1.08e+00  -1.7 2.36e-01    -  1.00e+00 1.00e+00f  1
   4  4.3223836e-02 0.00e+00 1.23e+00  -1.7 2.61e-01    -  1.00e+00 1.00e+00f  1
   5  1.5610508e-02 0.00e+00 3.54e-01  -1.7 1.18e-01    -  1.00e+00 1.00e+00f  1
   6  5.3544798e-03 0.00e+00 5.51e-01  -1.7 1.67e-01    -  1.00e+00 1.00e+00f  1
   7  6.1281576e-04 0.00e+00 5.19e-02  -1.7 3.87e-02    -  1.00e+00 1.00e+00f  1
   8  2.8893076e-05 0.00e+00 4.52e-02  -2.5 4.53e-02    -  1.00e+00 1.00e+00f  1
   9  3.4591761e-08 0.00e+00 3.80e-04  -2.5 3.18e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.2680803e-13 0.00e+00 3.02e-06  -5.7 3.62e-04    -  1.00e+00 1.00e+00f  1
  11  7.0136460e-25 0.00e+00 1.72e-12  -8.6 2.13e-07    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 11

                                   (scaled)                 (unscaled)
Objective...............:   1.5551321399859192e-25    7.0136459513364959e-25
Dual infeasibility......:   1.7239720368203862e-12    7.7751138860599418e-12
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   1.7239720368203862e-12    7.7751138860599418e-12


Number of objective function evaluations             = 18
Number of objective gradient evaluations             = 12
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 11
Total CPU secs in IPOPT (w/o function evaluations)   =      0.000
Total CPU secs in NLP function evaluations           =      0.000

EXIT: Optimal Solution Found.

        """
        parsed_output = ipopt.Ipopt()._parse_ipopt_output(output)
        self.assertEqual(
            {
                'iters': 11,
                'iteration_log': [
                    {
                        'iter': 0,
                        'objective': 56.5,
                        'inf_pr': 0.0,
                        'inf_du': 100.0,
                        'lg_mu': -1.0,
                        'd_norm': 0.0,
                        'lg_rg': None,
                        'alpha_du': 0.0,
                        'alpha_pr': 0.0,
                        'ls': 0,
                        'restoration': False,
                        'step_acceptance': None,
                    },
                    {
                        'iter': 1,
                        'objective': 0.24669972,
                        'inf_pr': 0.0,
                        'inf_du': 0.222,
                        'lg_mu': -1.0,
                        'd_norm': 0.74,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 2,
                        'objective': 0.16256267,
                        'inf_pr': 0.0,
                        'inf_du': 2.04,
                        'lg_mu': -1.7,
                        'd_norm': 1.48,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 0.25,
                        'ls': 3,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 3,
                        'objective': 0.086119444,
                        'inf_pr': 0.0,
                        'inf_du': 1.08,
                        'lg_mu': -1.7,
                        'd_norm': 0.236,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 4,
                        'objective': 0.043223836,
                        'inf_pr': 0.0,
                        'inf_du': 1.23,
                        'lg_mu': -1.7,
                        'd_norm': 0.261,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 5,
                        'objective': 0.015610508,
                        'inf_pr': 0.0,
                        'inf_du': 0.354,
                        'lg_mu': -1.7,
                        'd_norm': 0.118,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 6,
                        'objective': 0.0053544798,
                        'inf_pr': 0.0,
                        'inf_du': 0.551,
                        'lg_mu': -1.7,
                        'd_norm': 0.167,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 7,
                        'objective': 0.00061281576,
                        'inf_pr': 0.0,
                        'inf_du': 0.0519,
                        'lg_mu': -1.7,
                        'd_norm': 0.0387,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 8,
                        'objective': 2.8893076e-05,
                        'inf_pr': 0.0,
                        'inf_du': 0.0452,
                        'lg_mu': -2.5,
                        'd_norm': 0.0453,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 9,
                        'objective': 3.4591761e-08,
                        'inf_pr': 0.0,
                        'inf_du': 0.00038,
                        'lg_mu': -2.5,
                        'd_norm': 0.00318,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 10,
                        'objective': 1.2680803e-13,
                        'inf_pr': 0.0,
                        'inf_du': 3.02e-06,
                        'lg_mu': -5.7,
                        'd_norm': 0.000362,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 11,
                        'objective': 7.013646e-25,
                        'inf_pr': 0.0,
                        'inf_du': 1.72e-12,
                        'lg_mu': -8.6,
                        'd_norm': 2.13e-07,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                ],
                'incumbent_objective': 7.013645951336496e-25,
                'dual_infeasibility': 7.775113886059942e-12,
                'constraint_violation': 0.0,
                'complementarity_error': 0.0,
                'overall_nlp_error': 7.775113886059942e-12,
                'final_scaled_results': {
                    'incumbent_objective': 1.5551321399859192e-25,
                    'dual_infeasibility': 1.7239720368203862e-12,
                    'constraint_violation': 0.0,
                    'complementarity_error': 0.0,
                    'overall_nlp_error': 1.7239720368203862e-12,
                },
                'cpu_seconds': {
                    'IPOPT (w/o function evaluations)': 0.0,
                    'NLP function evaluations': 0.0,
                },
            },
            parsed_output,
        )

        # New ipopt style (3.14+)
        output = """******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit
******************************************************************************

This is Ipopt version 3.14.17, running with linear solver ma27.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:        3

Total number of variables............................:        2
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  5.6500000e+01 0.00e+00 1.00e+02  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.4669972e-01 0.00e+00 2.22e-01  -1.0 7.40e-01    -  1.00e+00 1.00e+00f  1
   2  1.6256267e-01 0.00e+00 2.04e+00  -1.7 1.48e+00    -  1.00e+00 2.50e-01f  3
   3  8.6119444e-02 0.00e+00 1.08e+00  -1.7 2.36e-01    -  1.00e+00 1.00e+00f  1
   4  4.3223836e-02 0.00e+00 1.23e+00  -1.7 2.61e-01    -  1.00e+00 1.00e+00f  1
   5  1.5610508e-02 0.00e+00 3.54e-01  -1.7 1.18e-01    -  1.00e+00 1.00e+00f  1
   6  5.3544798e-03 0.00e+00 5.51e-01  -1.7 1.67e-01    -  1.00e+00 1.00e+00f  1
   7  6.1281576e-04 0.00e+00 5.19e-02  -1.7 3.87e-02    -  1.00e+00 1.00e+00f  1
   8  2.8893076e-05 0.00e+00 4.52e-02  -2.5 4.53e-02    -  1.00e+00 1.00e+00f  1
   9  3.4591761e-08 0.00e+00 3.80e-04  -2.5 3.18e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.2680803e-13 0.00e+00 3.02e-06  -5.7 3.62e-04    -  1.00e+00 1.00e+00f  1
  11  7.0136460e-25 0.00e+00 1.72e-12  -8.6 2.13e-07    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 11

                                   (scaled)                 (unscaled)
Objective...............:   1.5551321399859192e-25    7.0136459513364959e-25
Dual infeasibility......:   1.7239720368203862e-12    7.7751138860599418e-12
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
Overall NLP error.......:   1.7239720368203862e-12    7.7751138860599418e-12


Number of objective function evaluations             = 18
Number of objective gradient evaluations             = 12
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 11
Total seconds in IPOPT                               = 0.002

EXIT: Optimal Solution Found.

Ipopt 3.14.17: Optimal Solution Found
        """
        parsed_output = ipopt.Ipopt()._parse_ipopt_output(output)
        self.assertEqual(
            {
                'iters': 11,
                'iteration_log': [
                    {
                        'iter': 0,
                        'objective': 56.5,
                        'inf_pr': 0.0,
                        'inf_du': 100.0,
                        'lg_mu': -1.0,
                        'd_norm': 0.0,
                        'lg_rg': None,
                        'alpha_du': 0.0,
                        'alpha_pr': 0.0,
                        'ls': 0,
                        'restoration': False,
                        'step_acceptance': None,
                    },
                    {
                        'iter': 1,
                        'objective': 0.24669972,
                        'inf_pr': 0.0,
                        'inf_du': 0.222,
                        'lg_mu': -1.0,
                        'd_norm': 0.74,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 2,
                        'objective': 0.16256267,
                        'inf_pr': 0.0,
                        'inf_du': 2.04,
                        'lg_mu': -1.7,
                        'd_norm': 1.48,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 0.25,
                        'ls': 3,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 3,
                        'objective': 0.086119444,
                        'inf_pr': 0.0,
                        'inf_du': 1.08,
                        'lg_mu': -1.7,
                        'd_norm': 0.236,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 4,
                        'objective': 0.043223836,
                        'inf_pr': 0.0,
                        'inf_du': 1.23,
                        'lg_mu': -1.7,
                        'd_norm': 0.261,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 5,
                        'objective': 0.015610508,
                        'inf_pr': 0.0,
                        'inf_du': 0.354,
                        'lg_mu': -1.7,
                        'd_norm': 0.118,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 6,
                        'objective': 0.0053544798,
                        'inf_pr': 0.0,
                        'inf_du': 0.551,
                        'lg_mu': -1.7,
                        'd_norm': 0.167,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 7,
                        'objective': 0.00061281576,
                        'inf_pr': 0.0,
                        'inf_du': 0.0519,
                        'lg_mu': -1.7,
                        'd_norm': 0.0387,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 8,
                        'objective': 2.8893076e-05,
                        'inf_pr': 0.0,
                        'inf_du': 0.0452,
                        'lg_mu': -2.5,
                        'd_norm': 0.0453,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 9,
                        'objective': 3.4591761e-08,
                        'inf_pr': 0.0,
                        'inf_du': 0.00038,
                        'lg_mu': -2.5,
                        'd_norm': 0.00318,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 10,
                        'objective': 1.2680803e-13,
                        'inf_pr': 0.0,
                        'inf_du': 3.02e-06,
                        'lg_mu': -5.7,
                        'd_norm': 0.000362,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 11,
                        'objective': 7.013646e-25,
                        'inf_pr': 0.0,
                        'inf_du': 1.72e-12,
                        'lg_mu': -8.6,
                        'd_norm': 2.13e-07,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                ],
                'incumbent_objective': 7.013645951336496e-25,
                'dual_infeasibility': 7.775113886059942e-12,
                'constraint_violation': 0.0,
                'variable_bound_violation': 0.0,
                'complementarity_error': 0.0,
                'overall_nlp_error': 7.775113886059942e-12,
                'final_scaled_results': {
                    'incumbent_objective': 1.5551321399859192e-25,
                    'dual_infeasibility': 1.7239720368203862e-12,
                    'constraint_violation': 0.0,
                    'variable_bound_violation': 0.0,
                    'complementarity_error': 0.0,
                    'overall_nlp_error': 1.7239720368203862e-12,
                },
                'cpu_seconds': {'IPOPT': 0.002},
            },
            parsed_output,
        )

    def test_empty_output_parsing(self):
        with self.assertLogs(
            "pyomo.contrib.solver.solvers.ipopt", level="WARNING"
        ) as logs:
            ipopt.Ipopt()._parse_ipopt_output(output=None)
        self.assertIn(
            "Returned output from ipopt was empty. Cannot parse for additional data.",
            logs.output[0],
        )

    def test_parse_output_diagnostic_tags(self):
        output = """******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit [legacy Ipopt URL removed]

This version of Ipopt was compiled from source code available at
    https://github.com/IDAES/Ipopt as part of the Institute for the Design of
    Advanced Energy Systems Process Systems Engineering Framework (IDAES PSE
    Framework) Copyright (c) 2018-2019. See https://github.com/IDAES/idaes-pse.

This version of Ipopt was compiled using HSL, a collection of Fortran codes
    for large-scale scientific computation.  All technical papers, sales and
    publicity material resulting from use of the HSL codes within IPOPT must
    contain the following acknowledgement:
        HSL, a collection of Fortran codes for large-scale scientific
        computation. See http://www.hsl.rl.ac.uk/.
******************************************************************************

This is Ipopt version 3.13.2, running with linear solver ma57.

Number of nonzeros in equality constraint Jacobian...:    77541
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    51855

Total number of variables............................:    15468
                     variables with only lower bounds:     3491
                variables with lower and upper bounds:     5026
                     variables with only upper bounds:      186
Total number of equality constraints.................:    15417
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.3126674e+00 1.34e+00 1.00e+00  -5.0 0.00e+00    -  0.00e+00 0.00e+00   0
Reallocating memory for MA57: lfact (2247250)
   1r 4.3126674e+00 1.34e+00 9.99e+02   0.1 0.00e+00  -4.0 0.00e+00 3.29e-10R  2
   2r 3.0519246e+08 1.13e+00 9.90e+02   0.1 2.30e+02    -  2.60e-02 9.32e-03f  1
   3r 2.2712595e+09 1.69e+00 9.73e+02   0.1 2.23e+02    -  2.54e-02 1.71e-02f  1 Nhj
   4  2.2712065e+09 1.69e+00 1.37e+09  -5.0 3.08e+03    -  1.32e-05 1.17e-05f  1 q
   5  1.9062986e+09 1.55e+00 1.25e+09  -5.0 5.13e+03    -  1.19e-01 8.38e-02f  1
   6  1.7041594e+09 1.46e+00 1.18e+09  -5.0 5.66e+03    -  7.06e-02 5.45e-02f  1
   7  1.4763158e+09 1.36e+00 1.10e+09  -5.0 3.94e+03    -  2.30e-01 6.92e-02f  1
   8  8.5873108e+08 1.04e+00 8.41e+08  -5.0 2.38e+05    -  3.49e-06 2.37e-01f  1
   9  4.4215572e+08 7.45e-01 6.03e+08  -5.0 1.63e+06    -  7.97e-02 2.82e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.0251884e+01 1.65e-01 1.57e+04  -5.0 1.24e+06    -  3.92e-05 1.00e+00f  1
  11  4.9121733e+01 4.97e-02 4.68e+03  -5.0 8.11e+04    -  4.31e-02 7.01e-01h  1
  12  4.1483985e+01 2.24e-02 5.97e+03  -5.0 1.15e+06    -  5.93e-02 1.00e+00f  1
  13  3.5762585e+01 1.75e-02 5.00e+03  -5.0 1.03e+06    -  1.25e-01 1.00e+00f  1
  14  3.2291014e+01 1.08e-02 3.51e+03  -5.0 8.25e+05    -  6.68e-01 1.00e+00f  1
  15  3.2274630e+01 3.31e-05 1.17e+00  -5.0 4.26e+04    -  9.92e-01 1.00e+00h  1
  16  3.2274631e+01 7.45e-09 2.71e-03  -5.0 6.11e+02    -  8.97e-01 1.00e+00h  1
  17  3.2274635e+01 7.45e-09 2.35e-03  -5.0 2.71e+04    -  1.32e-01 1.00e+00f  1
  18  3.2274635e+01 7.45e-09 1.15e-04  -5.0 5.53e+03    -  9.51e-01 1.00e+00h  1
  19  3.2274635e+01 7.45e-09 2.84e-05  -5.0 4.41e+04    -  7.54e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  3.2274635e+01 7.45e-09 8.54e-07  -5.0 1.83e+04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 20

                                   (scaled)                 (unscaled)
Objective...............:   3.2274635418964841e+01    3.2274635418964841e+01
Dual infeasibility......:   8.5365078678328669e-07    8.5365078678328669e-07
Constraint violation....:   8.0780625068607930e-13    7.4505805969238281e-09
Complementarity.........:   1.2275904566414160e-05    1.2275904566414160e-05
Overall NLP error.......:   1.2275904566414160e-05    1.2275904566414160e-05


Number of objective function evaluations             = 23
Number of objective gradient evaluations             = 20
Number of equality constraint evaluations            = 23
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 20
Total CPU secs in IPOPT (w/o function evaluations)   =     10.450
Total CPU secs in NLP function evaluations           =      1.651

EXIT: Optimal Solution Found.
    """
        parsed_output = ipopt.Ipopt()._parse_ipopt_output(output)
        self.assertEqual(
            {
                'iters': 20,
                'iteration_log': [
                    {
                        'iter': 0,
                        'objective': 4.3126674,
                        'inf_pr': 1.34,
                        'inf_du': 1.0,
                        'lg_mu': -5.0,
                        'd_norm': 0.0,
                        'lg_rg': None,
                        'alpha_du': 0.0,
                        'alpha_pr': 0.0,
                        'ls': 0,
                        'restoration': False,
                        'step_acceptance': None,
                    },
                    {
                        'iter': 1,
                        'objective': 4.3126674,
                        'inf_pr': 1.34,
                        'inf_du': 999.0,
                        'lg_mu': 0.1,
                        'd_norm': 0.0,
                        'lg_rg': -4.0,
                        'alpha_du': 0.0,
                        'alpha_pr': 3.29e-10,
                        'ls': 2,
                        'restoration': True,
                        'step_acceptance': 'R',
                    },
                    {
                        'iter': 2,
                        'objective': 305192460.0,
                        'inf_pr': 1.13,
                        'inf_du': 990.0,
                        'lg_mu': 0.1,
                        'd_norm': 230.0,
                        'lg_rg': None,
                        'alpha_du': 0.026,
                        'alpha_pr': 0.00932,
                        'ls': 1,
                        'restoration': True,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 3,
                        'objective': 2271259500.0,
                        'inf_pr': 1.69,
                        'inf_du': 973.0,
                        'lg_mu': 0.1,
                        'd_norm': 223.0,
                        'lg_rg': None,
                        'alpha_du': 0.0254,
                        'alpha_pr': 0.0171,
                        'ls': 1,
                        'restoration': True,
                        'step_acceptance': 'f',
                        'diagnostic_tags': 'Nhj',
                    },
                    {
                        'iter': 4,
                        'objective': 2271206500.0,
                        'inf_pr': 1.69,
                        'inf_du': 1370000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 3080.0,
                        'lg_rg': None,
                        'alpha_du': 1.32e-05,
                        'alpha_pr': 1.17e-05,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                        'diagnostic_tags': 'q',
                    },
                    {
                        'iter': 5,
                        'objective': 1906298600.0,
                        'inf_pr': 1.55,
                        'inf_du': 1250000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 5130.0,
                        'lg_rg': None,
                        'alpha_du': 0.119,
                        'alpha_pr': 0.0838,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 6,
                        'objective': 1704159400.0,
                        'inf_pr': 1.46,
                        'inf_du': 1180000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 5660.0,
                        'lg_rg': None,
                        'alpha_du': 0.0706,
                        'alpha_pr': 0.0545,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 7,
                        'objective': 1476315800.0,
                        'inf_pr': 1.36,
                        'inf_du': 1100000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 3940.0,
                        'lg_rg': None,
                        'alpha_du': 0.23,
                        'alpha_pr': 0.0692,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 8,
                        'objective': 858731080.0,
                        'inf_pr': 1.04,
                        'inf_du': 841000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 238000.0,
                        'lg_rg': None,
                        'alpha_du': 3.49e-06,
                        'alpha_pr': 0.237,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 9,
                        'objective': 442155720.0,
                        'inf_pr': 0.745,
                        'inf_du': 603000000.0,
                        'lg_mu': -5.0,
                        'd_norm': 1630000.0,
                        'lg_rg': None,
                        'alpha_du': 0.0797,
                        'alpha_pr': 0.282,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 10,
                        'objective': 50.251884,
                        'inf_pr': 0.165,
                        'inf_du': 15700.0,
                        'lg_mu': -5.0,
                        'd_norm': 1240000.0,
                        'lg_rg': None,
                        'alpha_du': 3.92e-05,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 11,
                        'objective': 49.121733,
                        'inf_pr': 0.0497,
                        'inf_du': 4680.0,
                        'lg_mu': -5.0,
                        'd_norm': 81100.0,
                        'lg_rg': None,
                        'alpha_du': 0.0431,
                        'alpha_pr': 0.701,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                    {
                        'iter': 12,
                        'objective': 41.483985,
                        'inf_pr': 0.0224,
                        'inf_du': 5970.0,
                        'lg_mu': -5.0,
                        'd_norm': 1150000.0,
                        'lg_rg': None,
                        'alpha_du': 0.0593,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 13,
                        'objective': 35.762585,
                        'inf_pr': 0.0175,
                        'inf_du': 5000.0,
                        'lg_mu': -5.0,
                        'd_norm': 1030000.0,
                        'lg_rg': None,
                        'alpha_du': 0.125,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 14,
                        'objective': 32.291014,
                        'inf_pr': 0.0108,
                        'inf_du': 3510.0,
                        'lg_mu': -5.0,
                        'd_norm': 825000.0,
                        'lg_rg': None,
                        'alpha_du': 0.668,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 15,
                        'objective': 32.27463,
                        'inf_pr': 3.31e-05,
                        'inf_du': 1.17,
                        'lg_mu': -5.0,
                        'd_norm': 42600.0,
                        'lg_rg': None,
                        'alpha_du': 0.992,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                    {
                        'iter': 16,
                        'objective': 32.274631,
                        'inf_pr': 7.45e-09,
                        'inf_du': 0.00271,
                        'lg_mu': -5.0,
                        'd_norm': 611.0,
                        'lg_rg': None,
                        'alpha_du': 0.897,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                    {
                        'iter': 17,
                        'objective': 32.274635,
                        'inf_pr': 7.45e-09,
                        'inf_du': 0.00235,
                        'lg_mu': -5.0,
                        'd_norm': 27100.0,
                        'lg_rg': None,
                        'alpha_du': 0.132,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 18,
                        'objective': 32.274635,
                        'inf_pr': 7.45e-09,
                        'inf_du': 0.000115,
                        'lg_mu': -5.0,
                        'd_norm': 5530.0,
                        'lg_rg': None,
                        'alpha_du': 0.951,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                    {
                        'iter': 19,
                        'objective': 32.274635,
                        'inf_pr': 7.45e-09,
                        'inf_du': 2.84e-05,
                        'lg_mu': -5.0,
                        'd_norm': 44100.0,
                        'lg_rg': None,
                        'alpha_du': 0.754,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 20,
                        'objective': 32.274635,
                        'inf_pr': 7.45e-09,
                        'inf_du': 8.54e-07,
                        'lg_mu': -5.0,
                        'd_norm': 18300.0,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                ],
                'incumbent_objective': 32.27463541896484,
                'dual_infeasibility': 8.536507867832867e-07,
                'constraint_violation': 7.450580596923828e-09,
                'complementarity_error': 1.227590456641416e-05,
                'overall_nlp_error': 1.227590456641416e-05,
                'final_scaled_results': {
                    'incumbent_objective': 32.27463541896484,
                    'dual_infeasibility': 8.536507867832867e-07,
                    'constraint_violation': 8.078062506860793e-13,
                    'complementarity_error': 1.227590456641416e-05,
                    'overall_nlp_error': 1.227590456641416e-05,
                },
                'cpu_seconds': {
                    'IPOPT (w/o function evaluations)': 10.45,
                    'NLP function evaluations': 1.651,
                },
            },
            parsed_output,
        )

    def test_parse_output_errors(self):
        output = """******************************************************************************
******************************************************************************

This is Ipopt version 3.13.2, running with linear solver ma57.

Number of nonzeros in equality constraint Jacobian...:    77541
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    51855

Total number of variables............................:    15468
                     variables with only lower bounds:     3491
                variables with lower and upper bounds:     5026
                     variables with only upper bounds:      186
Total number of equality constraints.................:    15417
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  4.3126674e+00 1.34e+00 1.00e+00  -5.0 0.00e+00    -  0.00e+00 0.00e+00   0
Reallocating memory for MA57: lfact (2247250)
   1r-4.3126674e+00 1.34e+00 9.99e+02   0.1 0.00e+00  -4.0 0.00e+00 3.29e-10R  2
  19t  3.2274635e+01 7.45e-09 2.84e-05  -5.0 4.41e+04    -  7.54e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  3.2274635e+01f 7.45e-09 8.54e-07  -5.0 1.83e+04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 20

                                   (scaled)                 (unscaled)
Objective...............:   3.2274635418964841e+01    3.2274635418964841e+01
Dual infeasibility......:   8.5365078678328669e-07    8.5365078678328669e-07
Constraint violation....:   8.0780625068607930e-13    7.4505805969238281e-09
Complementarity.........:   1.2275904566414160e-05    1.2275904566414160e-05
Overall NLP error.......:   1.2275904566414160e-05    1.2275904566414160e-05


Number of objective function evaluations             = 23
Number of objective gradient evaluations             = 20
Number of equality constraint evaluations            = 23
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 22
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 20
Total CPU secs in IPOPT (w/o function evaluations)   =     10.450
Total CPU secs in NLP function evaluations           =      1.651

EXIT: Optimal Solution Found.
    """
        with LoggingIntercept() as LOG:
            parsed_output = ipopt.Ipopt()._parse_ipopt_output(output)
        self.assertEqual(
            """Error parsing Ipopt log entry:
\tinvalid literal for int() with base 10: '19t'
\t  19t  3.2274635e+01 7.45e-09 2.84e-05  -5.0 4.41e+04    -  7.54e-01 1.00e+00f  1
Error parsing Ipopt log entry:
\tcould not convert string to float: '3.2274635e+01f'
\t  20  3.2274635e+01f 7.45e-09 8.54e-07  -5.0 1.83e+04    -  1.00e+00 1.00e+00h  1
Total number of iteration records parsed 4 does not match the number of iterations (20) plus one.
""",
            LOG.getvalue(),
        )
        self.assertEqual(
            {
                'iters': 20,
                'iteration_log': [
                    {
                        'iter': 0,
                        'objective': 4.3126674,
                        'inf_pr': 1.34,
                        'inf_du': 1.0,
                        'lg_mu': -5.0,
                        'd_norm': 0.0,
                        'lg_rg': None,
                        'alpha_du': 0.0,
                        'alpha_pr': 0.0,
                        'ls': 0,
                        'restoration': False,
                        'step_acceptance': None,
                    },
                    {
                        'iter': 1,
                        'objective': -4.3126674,
                        'inf_pr': 1.34,
                        'inf_du': 999.0,
                        'lg_mu': 0.1,
                        'd_norm': 0.0,
                        'lg_rg': -4.0,
                        'alpha_du': 0.0,
                        'alpha_pr': 3.29e-10,
                        'ls': 2,
                        'restoration': True,
                        'step_acceptance': 'R',
                    },
                    {
                        'iter': '19t',
                        'objective': 32.274635,
                        'inf_pr': 7.45e-09,
                        'inf_du': 2.84e-05,
                        'lg_mu': -5.0,
                        'd_norm': 44100.0,
                        'lg_rg': None,
                        'alpha_du': 0.754,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'f',
                    },
                    {
                        'iter': 20,
                        'objective': '3.2274635e+01f',
                        'inf_pr': 7.45e-09,
                        'inf_du': 8.54e-07,
                        'lg_mu': -5.0,
                        'd_norm': 18300.0,
                        'lg_rg': None,
                        'alpha_du': 1.0,
                        'alpha_pr': 1.0,
                        'ls': 1,
                        'restoration': False,
                        'step_acceptance': 'h',
                    },
                ],
                'incumbent_objective': 32.27463541896484,
                'dual_infeasibility': 8.536507867832867e-07,
                'constraint_violation': 7.450580596923828e-09,
                'complementarity_error': 1.227590456641416e-05,
                'overall_nlp_error': 1.227590456641416e-05,
                'final_scaled_results': {
                    'incumbent_objective': 32.27463541896484,
                    'dual_infeasibility': 8.536507867832867e-07,
                    'constraint_violation': 8.078062506860793e-13,
                    'complementarity_error': 1.227590456641416e-05,
                    'overall_nlp_error': 1.227590456641416e-05,
                },
                'cpu_seconds': {
                    'IPOPT (w/o function evaluations)': 10.45,
                    'NLP function evaluations': 1.651,
                },
            },
            parsed_output,
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_has_linear_solver(self):
        opt = ipopt.Ipopt()
        self.assertTrue(
            any(
                map(
                    opt.has_linear_solver,
                    [
                        'mumps',
                        'ma27',
                        'ma57',
                        'ma77',
                        'ma86',
                        'ma97',
                        'pardiso',
                        'pardisomkl',
                        'spral',
                        'wsmp',
                    ],
                )
            )
        )
        self.assertFalse(opt.has_linear_solver('bogus_linear_solver'))

    @unittest.skipIf(sys.platform.startswith("win"), "Test requires *nix")
    def test_command_line(self):
        with TempfileManager.new_context() as tempfile:
            dname = tempfile.mkdtemp()
            exe = os.path.join(dname, 'mock')
            with open(exe, 'w') as F:
                F.write(f"""#!{sys.executable}
import sys
if sys.argv[1] == '--version':
    print('ipopt 1.2.3 ASL')
else:
    print('\\n'.join(sys.argv[1:]))
    sys.exit(1)
""")
            os.chmod(exe, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            opt = ipopt.Ipopt(executable=exe)

            m = pyo.ConcreteModel()
            m.x = pyo.Var()
            m.o = pyo.Objective(expr=m.x**2)

            opts = dict(
                raise_exception_on_nonoptimal_result=False, load_solutions=False
            )
            # No custom options, no file created. Plain and simple.
            with LoggingIntercept() as LOG:
                result = opt.solve(m, **opts)
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL'])

            # Custom command line options
            with LoggingIntercept() as LOG:
                result = opt.solve(m, **opts, solver_options={'max_iter': 4})
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_iter=4'])

            # Custom command line options; threads generates a warning
            with LoggingIntercept() as LOG:
                result = opt.solve(
                    m, **opts, threads=10, solver_options={'max_iter': 4}
                )
            self.assertEqual(
                "The `threads=10` option was specified, "
                "but this is not used by Ipopt.\n",
                LOG.getvalue(),
            )
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_iter=4'])

            # Custom command line options; threads generates a warning... unless it's 1
            with LoggingIntercept() as LOG:
                result = opt.solve(m, **opts, threads=1, solver_options={'max_iter': 4})
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_iter=4'])

            # Let's see if we correctly parse config.time_limit
            with LoggingIntercept() as LOG:
                result = opt.solve(
                    m, **opts, time_limit=10, solver_options={'max_iter': 4}
                )
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_iter=4', 'max_cpu_time=10.0'])

            # Now let's do multiple command line options
            with LoggingIntercept() as LOG:
                result = opt.solve(
                    m, **opts, solver_options={'max_iter': 4, 'max_cpu_time': 20}
                )
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_cpu_time=20', 'max_iter=4'])

            # but top-level options override solver_options
            with LoggingIntercept() as LOG:
                result = opt.solve(
                    m,
                    **opts,
                    time_limit=10,
                    solver_options={'max_iter': 4, 'max_cpu_time': 20},
                )
            self.assertEqual("", LOG.getvalue())
            cmd = result.solver_log.splitlines()
            self.assertTrue(cmd[0].endswith('nl'))
            self.assertEqual(cmd[1:], ['-AMPL', 'max_cpu_time=10.0', 'max_iter=4'])

    def test_option_to_str(self):
        # int / float / str
        self.assertEqual('opt=5', ipopt._option_to_cmd('opt', 5))
        self.assertEqual('opt=5.0', ipopt._option_to_cmd('opt', 5.0))
        self.assertEqual('opt="5"', ipopt._option_to_cmd('opt', '5'))

        # If the string contains a quote, then the name needs to be
        # quoted
        self.assertEqual("opt=\"'model'\"", ipopt._option_to_cmd('opt', "'model'"))
        self.assertEqual("opt='\"model\"'", ipopt._option_to_cmd('opt', '"model"'))
        # but if it has both, we will error
        with self.assertRaisesRegex(ValueError, 'single and double'):
            ipopt._option_to_cmd('opt', '"\'model"')

    def test_process_options(self):
        solver = ipopt.Ipopt()
        with TempfileManager.new_context() as TMP:
            dname = TMP.mkdtemp()

            # test no options
            fname = os.path.join(dname, 'test1.txt')
            cmd = solver._process_options(fname, {})
            self.assertFalse(os.path.exists(fname))
            self.assertEqual([], cmd)

            # command-line only options
            fname = os.path.join(dname, 'test2.txt')
            cmd = solver._process_options(fname, {'bound_push': 'no', 'max_iter': 5})
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(['bound_push="no"', 'max_iter=5'], cmd)

            # both command line and options file
            fname = os.path.join(dname, 'test3.txt')
            cmd = solver._process_options(
                fname, {'custom_option_2': 5, 'bound_push': 'no', 'custom_option_1': 3}
            )
            self.assertTrue(os.path.exists(fname))
            with open(fname, 'r') as F:
                self.assertEqual('custom_option_2 5\ncustom_option_1 3\n', F.read())
            if '"' in fname:
                fname = "'" + fname + "'"
            else:
                fname = '"' + fname + '"'
            self.assertEqual(['bound_push="no"', f'option_file_name={fname}'], cmd)

            # only options file
            fname = os.path.join(dname, 'test4.txt')
            cmd = solver._process_options(fname, {'custom_option_3': 3})
            self.assertTrue(os.path.exists(fname))
            with open(fname, 'r') as F:
                self.assertEqual('custom_option_3 3\n', F.read())
            if '"' in fname:
                fname = "'" + fname + "'"
            else:
                fname = '"' + fname + '"'
            self.assertEqual([f'option_file_name={fname}'], cmd)

            # illegal options
            fname = os.path.join(dname, 'test5.txt')
            with self.assertRaisesRegex(
                ValueError,
                "unallowed Ipopt option 'wantsol': "
                "The solver interface requires the sol file to be created",
            ):
                solver._process_options(fname, {'bogus': 3, 'wantsol': False})
            self.assertFalse(os.path.exists(fname))

            fname = os.path.join(dname, 'test5.txt')
            with self.assertRaisesRegex(
                ValueError,
                "unallowed Ipopt option 'option_file_name': "
                'Pyomo generates the ipopt options file as part of the `solve` '
                'method.  Add all options to config.solver_options instead',
            ):
                solver._process_options(
                    fname, {'bogus': 3, 'option_file_name': 'myfile.opt'}
                )
            self.assertFalse(os.path.exists(fname))

    def test_presolve_prove_infeasible(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x == 10)
        m.obj = pyo.Objective(expr=m.x)

        timer = HierarchicalTimer()
        solver = ipopt.Ipopt()
        results = solver.solve(
            m,
            timer=timer,
            load_solutions=False,
            raise_exception_on_nonoptimal_result=False,
        )
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertEqual(
            results.termination_condition, TerminationCondition.provenInfeasible
        )
        cfg = results.solver_config
        del cfg.executable
        self.assertEqual(
            {
                'load_solutions': False,
                'raise_exception_on_nonoptimal_result': False,
                'solver_options': {},
                'symbolic_solver_labels': False,
                'tee': [],
                'threads': None,
                'time_limit': None,
                'timer': timer,
                'working_dir': None,
                'writer_config': {
                    'column_order': None,
                    'export_defined_variables': True,
                    'export_nonlinear_variables': None,
                    'file_determinism': FileDeterminism.ORDERED,
                    'linear_presolve': True,
                    'row_order': None,
                    'scale_model': True,
                    'show_section_timing': False,
                    'skip_trivial_constraints': True,
                    'symbolic_solver_labels': False,
                },
            },
            cfg.value(),
        )
        self.assertLess(results.timing_info.wall_time, 1)
        self.assertEqual(
            results.timing_info.start_timestamp.tzinfo, datetime.timezone.utc
        )
        self.assertLess(
            (
                datetime.datetime.now(datetime.timezone.utc)
                - results.timing_info.start_timestamp
            ).seconds,
            1,
        )
        del results.extra_info.base_file_name
        del results.solver_config
        del results.timing_info.wall_time
        del results.timing_info.start_timestamp
        self.assertEqual(
            {
                'extra_info': {'iteration_count': 0},
                'incumbent_objective': None,
                'objective_bound': None,
                'solution_loader': None,
                'solution_status': SolutionStatus.noSolution,
                'solver_log': None,
                'solver_name': 'ipopt',
                'solver_version': None,
                'termination_condition': TerminationCondition.provenInfeasible,
                'timing_info': {'timer': timer},
            },
            results.value(),
        )

        with self.assertRaisesRegex(
            NoSolutionError, "Solution loader does not currently have a valid solution."
        ):
            results = solver.solve(
                m, timer=timer, raise_exception_on_nonoptimal_result=False
            )
        with self.assertRaisesRegex(
            NoOptimalSolutionError, "Solver did not find the optimal solution."
        ):
            results = solver.solve(m, timer=timer)

    def test_presolve_solveModel(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 50))
        m.c = pyo.Constraint(expr=m.x == 10)
        m.obj = pyo.Objective(expr=m.x)

        timer = HierarchicalTimer()
        solver = ipopt.Ipopt()
        results = solver.solve(m, timer=timer)
        self.assertEqual(results.solution_status, SolutionStatus.optimal)
        self.assertEqual(
            results.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        cfg = results.solver_config
        del results.solver_config
        del cfg.executable
        self.assertEqual(
            {
                'load_solutions': True,
                'raise_exception_on_nonoptimal_result': True,
                'solver_options': {},
                'symbolic_solver_labels': False,
                'tee': [],
                'threads': None,
                'time_limit': None,
                'timer': timer,
                'working_dir': None,
                'writer_config': {
                    'column_order': None,
                    'export_defined_variables': True,
                    'export_nonlinear_variables': None,
                    'file_determinism': FileDeterminism.ORDERED,
                    'linear_presolve': True,
                    'row_order': None,
                    'scale_model': True,
                    'show_section_timing': False,
                    'skip_trivial_constraints': True,
                    'symbolic_solver_labels': False,
                },
            },
            cfg.value(),
        )
        self.assertLess(results.timing_info.wall_time, 1)
        del results.timing_info.wall_time
        self.assertEqual(
            results.timing_info.start_timestamp.tzinfo, datetime.timezone.utc
        )
        self.assertLess(
            (
                datetime.datetime.now(datetime.timezone.utc)
                - results.timing_info.start_timestamp
            ).seconds,
            1,
        )
        del results.timing_info.start_timestamp
        del results.extra_info.base_file_name
        self.assertIsNotNone(results.solution_loader)
        del results.solution_loader
        self.assertEqual(
            {
                'extra_info': {'iteration_count': 0},
                'incumbent_objective': 10.0,
                'objective_bound': None,
                'solution_status': SolutionStatus.optimal,
                'solver_log': None,
                'solver_name': 'ipopt',
                'solver_version': None,
                'termination_condition': TerminationCondition.convergenceCriteriaSatisfied,
                'timing_info': {'timer': timer},
            },
            results.value(),
        )
        self.assertEqual(m.x.value, 10)

    def test_presolve_empty(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=1)

        timer = HierarchicalTimer()
        solver = ipopt.Ipopt()
        results = solver.solve(
            m,
            timer=timer,
            load_solutions=False,
            raise_exception_on_nonoptimal_result=False,
        )
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertEqual(results.termination_condition, TerminationCondition.emptyModel)
        cfg = results.solver_config
        del cfg.executable
        self.assertEqual(
            {
                'load_solutions': False,
                'raise_exception_on_nonoptimal_result': False,
                'solver_options': {},
                'symbolic_solver_labels': False,
                'tee': [],
                'threads': None,
                'time_limit': None,
                'timer': timer,
                'working_dir': None,
                'writer_config': {
                    'column_order': None,
                    'export_defined_variables': True,
                    'export_nonlinear_variables': None,
                    'file_determinism': FileDeterminism.ORDERED,
                    'linear_presolve': True,
                    'row_order': None,
                    'scale_model': True,
                    'show_section_timing': False,
                    'skip_trivial_constraints': True,
                    'symbolic_solver_labels': False,
                },
            },
            cfg.value(),
        )
        self.assertLess(results.timing_info.wall_time, 1)
        self.assertEqual(
            results.timing_info.start_timestamp.tzinfo, datetime.timezone.utc
        )
        self.assertLess(
            (
                datetime.datetime.now(datetime.timezone.utc)
                - results.timing_info.start_timestamp
            ).seconds,
            1,
        )
        del results.extra_info.base_file_name
        del results.solver_config
        del results.timing_info.wall_time
        del results.timing_info.start_timestamp
        self.assertEqual(
            {
                'extra_info': {'iteration_count': 0},
                'incumbent_objective': None,
                'objective_bound': None,
                'solution_loader': None,
                'solution_status': SolutionStatus.noSolution,
                'solver_log': None,
                'solver_name': 'ipopt',
                'solver_version': None,
                'termination_condition': TerminationCondition.emptyModel,
                'timing_info': {'timer': timer},
            },
            results.value(),
        )

        with self.assertRaisesRegex(
            NoSolutionError, "Solution loader does not currently have a valid solution."
        ):
            results = solver.solve(
                m, timer=timer, raise_exception_on_nonoptimal_result=False
            )
        with self.assertRaisesRegex(
            NoOptimalSolutionError, "Solver did not find the optimal solution."
        ):
            results = solver.solve(m, timer=timer)

    def test_file_collision(self):
        class mock_tempfile:
            def __init__(self):
                self.fd = None

            def new_context(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if self.fd is not None:
                    os.close(self.fd)

            def mkstemp(self, suffix, prefix, dir, text, delete):
                fname = os.path.join(dir, "testfile" + suffix)
                self.fd = os.open(fname, os.O_CREAT | os.O_RDWR)
                return self.fd, fname

        m = pyo.ConcreteModel()
        orig_TempfileManager = ipopt.TempfileManager
        try:
            ipopt.TempfileManager = mock_tempfile()
            with TempfileManager.new_context() as tempfile:
                solver = ipopt.Ipopt()
                dname = tempfile.mkdtemp()
                for ext in ('.row', '.col', '.sol', '.opt'):
                    fname = os.path.join(dname, 'testfile' + ext)
                    open(fname, 'w').close()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        f"Solver interface file "
                        + fname.replace('\\', '\\\\')
                        + " already exists",
                    ):
                        solver.solve(m, working_dir=dname)
                    os.unlink(fname)
        finally:
            ipopt.TempfileManager = orig_TempfileManager

    def test_bad_executable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.o = pyo.Objective(expr=m.x**2)

        solver = ipopt.Ipopt()

        with TempfileManager.new_context() as tempfile:
            dname = tempfile.mkdtemp()
            exe = os.path.join(dname, 'ipopt')
            solver.config.executable = exe
            with self.assertRaisesRegex(ApplicationError, 'ipopt executable not found'):
                solver.solve(m)

            # The following is designed to run on *NIX
            if sys.platform.startswith("win"):
                return

            _cache = ipopt.Ipopt._exe_cache
            ipopt.Ipopt._exe_cache = {exe: (1, 2, 3)}
            try:
                with open(exe, 'w') as F:
                    F.write(f"#!{dname}/bad_interpreter\nsys.exit(1)\n")
                os.chmod(exe, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                solver.config.executable.rehash()
                with self.assertRaisesRegex(
                    ApplicationError,
                    f"Could not execute the command: \\['{exe}'.*"
                    f"Error message: .*No such file or directory: '{exe}'",
                ):
                    solver.solve(m)
            finally:
                ipopt.Ipopt._exe_cache = _cache


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.pytest.mark.solver("ipopt")
class TestIpopt(unittest.TestCase):
    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5)
        model.y = pyo.Var(initialize=1.5)

        def rosenbrock(m):
            return (1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2

        model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)
        return model

    def test_ipopt_config(self):
        # Test default initialization
        config = ipopt.IpoptConfig()
        self.assertTrue(config.load_solutions)
        self.assertIsInstance(config.solver_options, ConfigDict)
        self.assertIsInstance(config.executable, ExecutableData)

        # Test custom initialization
        solver = SolverFactory('ipopt', executable='/path/to/exe')
        self.assertFalse(solver.config.tee)
        self.assertIsNone(solver.config.executable.path())
        self.assertTrue(solver.config.executable._registered_name.startswith('/path'))

    def test_ipopt_solve(self):
        # Gut check - does it solve?
        model = self.create_model()
        ipopt.Ipopt().solve(model)
        self.assertAlmostEqual(model.x.value, 1)
        self.assertAlmostEqual(model.y.value, 1)

    def test_ipopt_quiet_print_level(self):
        model = self.create_model()
        result = ipopt.Ipopt().solve(model, solver_options={'print_level': 0})
        # IPOPT doesn't tell us anything about the iters if the print level
        # is set to 0
        self.assertEqual(result.extra_info.iteration_count, None)
        self.assertFalse(hasattr(result.extra_info, 'iteration_log'))
        model = self.create_model()
        result = ipopt.Ipopt().solve(model, solver_options={'print_level': 3})
        # At a slightly higher level, we get some of the info, like
        # iteration count, but NOT iteration_log
        self.assertEqual(result.extra_info.iteration_count, 11)
        self.assertFalse(hasattr(result.extra_info, 'iteration_log'))

    def test_ipopt_loud_print_level(self):
        with windows_tee_buffer(1 << 20):
            model = self.create_model()
            result = ipopt.Ipopt().solve(model, solver_options={'print_level': 8})
            # Nothing unexpected should be in the results object at this point,
            # except that the solver_log is significantly longer
            self.assertEqual(result.extra_info.iteration_count, 11)
            self.assertEqual(result.incumbent_objective, 7.013645951336496e-25)
            self.assertIn('Optimal Solution Found', result.extra_info.solver_message)
            self.assertTrue(hasattr(result.extra_info, 'iteration_log'))
            model = self.create_model()
            result = ipopt.Ipopt().solve(model, solver_options={'print_level': 12})
            self.assertEqual(result.extra_info.iteration_count, 11)
            self.assertEqual(result.incumbent_objective, 7.013645951336496e-25)
            self.assertIn('Optimal Solution Found', result.extra_info.solver_message)
            self.assertTrue(hasattr(result.extra_info, 'iteration_log'))

    def test_ipopt_results(self):
        model = self.create_model()
        results = ipopt.Ipopt().solve(model)
        self.assertEqual(results.solver_name, 'ipopt')
        self.assertEqual(results.extra_info.iteration_count, 11)
        self.assertEqual(results.incumbent_objective, 7.013645951336496e-25)
        self.assertIn('Optimal Solution Found', results.extra_info.solver_message)

    def test_ipopt_results_display(self):
        model = self.create_model()
        results = ipopt.Ipopt().solve(model)
        # Do not show extra loud stuff
        with capture_output() as OUT:
            results.display()
        contents = OUT.getvalue()
        self.assertIn('termination_condition', contents)
        self.assertIn('solution_status', contents)
        self.assertIn('incumbent_objective', contents)
        self.assertNotIn('iteration_log', contents)
        # Now we want to see the iteration log
        with capture_output() as OUT:
            results.display(visibility=ADVANCED_OPTION)
        contents = OUT.getvalue()
        self.assertIn('termination_condition', contents)
        self.assertIn('solution_status', contents)
        self.assertIn('incumbent_objective', contents)
        self.assertIn('iteration_log', contents)

    def test_ipopt_timer_object(self):
        model = self.create_model()
        ipopt_instance = ipopt.Ipopt()
        results = ipopt_instance.solve(model)
        timing_info = results.timing_info
        if ipopt_instance.version()[0:1] <= (3, 13):
            # We are running an older version of IPOPT (<= 3.13)
            self.assertIn('IPOPT (w/o function evaluations)', timing_info.keys())
            self.assertIn('NLP function evaluations', timing_info.keys())
        else:
            # Newer version of IPOPT
            self.assertIn('IPOPT', timing_info.keys())

    def test_run_ipopt_options_file(self):
        # Check that the options file is getting to Ipopt: if we give it
        # an invalid option in the options file, ipopt will fail.  This
        # is important, as ipopt will NOT fail if you pass if an
        # option_file_name that does not exist.
        model = self.create_model()
        results = ipopt.Ipopt().solve(
            model,
            solver_options={'bogus_option': 5},
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        self.assertEqual(results.termination_condition, TerminationCondition.error)
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertIn('OPTION_INVALID', results.solver_log)

        # Verify the full command line once
        cmd = results.extra_info.command_line
        fname = results.extra_info.base_file_name
        self.assertEqual(
            [
                str(ipopt.Ipopt().config.executable),
                f'{fname}.nl',
                "-AMPL",
                f'option_file_name="{fname}.opt"',
            ],
            cmd,
        )

    def test_ipopt_working_dir(self):
        m = self.create_model()
        with TempfileManager.new_context() as tempfile:
            dname = tempfile.mkdtemp()
            working_dir = os.path.join(dname, 'testing')
            self.assertFalse(os.path.exists(working_dir))

            results = ipopt.Ipopt().solve(m, working_dir=working_dir)

            self.assertTrue(os.path.exists(working_dir))
            self.assertTrue(results.extra_info.base_file_name.startswith(working_dir))
            self.assertTrue(os.path.exists(results.extra_info.base_file_name + '.nl'))

    def test_load_solution_suffixes(self):
        m = self.create_model()
        m.x.lb = 0.6
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.c = pyo.Constraint(expr=m.x == 2 * m.y)

        solver = ipopt.Ipopt()
        results = solver.solve(m, writer_config={'linear_presolve': False})
        o1 = results.extra_info.incumbent_objective

        self.assertAlmostEqual(m.x.value, 0.6, delta=1e-5)
        self.assertAlmostEqual(m.y.value, 0.3, delta=1e-5)
        self.assertEqual(len(m.dual), 1)
        self.assertAlmostEqual(m.dual[m.c], 6, delta=1e-5)
        self.assertEqual(len(m.rc), 2)
        self.assertAlmostEqual(m.rc[m.x], 7.6, delta=1e-5)
        self.assertEqual(m.rc[m.y], 0)

        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.obj] = 10
        m.scaling_factor[m.c] = 5
        m.scaling_factor[m.x] = 7
        m.scaling_factor[m.y] = 3
        results = solver.solve(m, writer_config={'linear_presolve': False})

        o2 = results.extra_info.incumbent_objective
        self.assertAlmostEqual(o1, o2 / 10, delta=1e-5)

        self.assertAlmostEqual(m.x.value, 0.6, delta=1e-5)
        self.assertAlmostEqual(m.y.value, 0.3, delta=1e-5)
        self.assertEqual(len(m.dual), 1)
        self.assertAlmostEqual(m.dual[m.c], 6, delta=1e-5)
        self.assertEqual(len(m.rc), 2)
        self.assertAlmostEqual(m.rc[m.x], 7.6, delta=1e-5)
        self.assertEqual(m.rc[m.y], 0)

        m.x.lb = None
        m.y.ub = 0.25
        results = solver.solve(m, writer_config={'linear_presolve': False})

        self.assertAlmostEqual(m.x.value, 0.5, delta=1e-5)
        self.assertAlmostEqual(m.y.value, 0.25, delta=1e-5)
        self.assertEqual(len(m.dual), 1)
        self.assertAlmostEqual(m.dual[m.c], -1, delta=1e-5)
        self.assertEqual(len(m.rc), 2)
        self.assertEqual(m.rc[m.x], 0)
        self.assertAlmostEqual(m.rc[m.y], -2, delta=1e-5)

    def test_load_suffixes_infeasible_model(self):
        # This tests Issue #3807: Ipopt was failing to load duals /
        # reduced coses when the solver exited in restoration
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2, bounds=(1, 10))
        m.c = pyo.Constraint(expr=(m.x == 0))

        solver = ipopt.Ipopt()
        solver.config.writer_config.linear_presolve = False
        solver.config.raise_exception_on_nonoptimal_result = False

        results = solver.solve(m)
        self.assertEqual(results.solution_status, SolutionStatus.infeasible)
        self.assertEqual(results.extra_info.iteration_log[-1]['restoration'], True)
        self.assertAlmostEqual(results.solution_loader.get_reduced_costs()[m.x], 1000)
        self.assertAlmostEqual(results.solution_loader.get_duals()[m.c], -1000)


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.pytest.mark.solver("ipopt")
class TestLegacyIpopt(unittest.TestCase):
    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5)
        model.y = pyo.Var(initialize=1.5)

        @model.Objective(sense=pyo.minimize)
        def rosenbrock(m):
            return (1.0 - m.x) ** 2 + 100.0 * (m.y - m.x**2) ** 2

        return model

    def test_map_OF_options(self):
        model = self.create_model()

        with capture_output() as LOG:
            results = ipopt.LegacyIpoptSolver().solve(
                model,
                tee=True,
                solver_options={'OF_bogus_option': 5},
                load_solutions=False,
            )
        self.assertIn('OPTION_INVALID', LOG.getvalue())
        # Note: OF_ is stripped
        self.assertIn(
            'Read Option: "bogus_option". It is not a valid option', LOG.getvalue()
        )

        with self.assertRaisesRegex(ValueError, "unallowed Ipopt option 'wantsol'"):
            results = ipopt.LegacyIpoptSolver().solve(
                model,
                tee=True,
                solver_options={'OF_wantsol': False},
                load_solutions=False,
            )
