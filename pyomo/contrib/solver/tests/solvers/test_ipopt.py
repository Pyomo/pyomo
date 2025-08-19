#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import subprocess

import pyomo.environ as pyo
from pyomo.common.envvar import is_windows
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict, ADVANCED_OPTION
from pyomo.common.errors import DeveloperError
from pyomo.common.tee import capture_output
import pyomo.contrib.solver.solvers.ipopt as ipopt
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.contrib.solver.common.results import TerminationCondition, SolutionStatus
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.common import unittest, Executable
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter

ipopt_available = ipopt.Ipopt().available()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
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
        self.assertIsNotNone(str(config.executable))
        self.assertIn('ipopt', str(config.executable))
        # Set to a totally bogus path
        config.executable = Executable('/bogus/path')
        self.assertIsNone(config.executable.executable)
        self.assertFalse(config.executable.available())


class TestIpoptSolutionLoader(unittest.TestCase):
    def test_get_reduced_costs_error(self):
        loader = ipopt.IpoptSolutionLoader(None, None)
        with self.assertRaises(NoSolutionError):
            loader.get_reduced_costs()

        # Set _nl_info to something completely bogus but is not None
        class NLInfo:
            pass

        loader._nl_info = NLInfo()
        loader._nl_info.eliminated_vars = [1, 2, 3]
        # This test may need to be altered if we enable returning duals
        # when presolve is on
        with self.assertRaises(NotImplementedError):
            loader.get_reduced_costs()
        # Reset _nl_info so we can ensure we get an error
        # when _sol_data is None
        loader._nl_info.eliminated_vars = []
        with self.assertRaises(DeveloperError):
            loader.get_reduced_costs()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):
    def test_command_line_options(self):
        result = subprocess.run(
            ['ipopt', '-='], capture_output=True, text=True, check=True
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
        method_list = [method for method in dir(opt) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = ipopt.Ipopt()
        self.assertFalse(opt.is_persistent())
        self.assertIsNotNone(opt.version())
        self.assertEqual(opt.name, 'ipopt')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())

    def test_context_manager(self):
        with ipopt.Ipopt() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertIsNotNone(opt.version())
            self.assertEqual(opt.name, 'ipopt')
            self.assertEqual(opt.CONFIG, opt.config)
            self.assertTrue(opt.available())

    def test_available_cache(self):
        opt = ipopt.Ipopt()
        opt.available()
        self.assertTrue(opt._available_cache[1])
        self.assertIsNotNone(opt._available_cache[0])
        # Now we will try with a custom config that has a fake path
        config = ipopt.IpoptConfig()
        config.executable = Executable('/a/bogus/path')
        opt.available(config=config)
        self.assertFalse(opt._available_cache[1])
        self.assertIsNone(opt._available_cache[0])

    def test_version_cache(self):
        opt = ipopt.Ipopt()
        opt.version()
        self.assertIsNotNone(opt._version_cache[0])
        self.assertIsNotNone(opt._version_cache[1])
        # Now we will try with a custom config that has a fake path
        config = ipopt.IpoptConfig()
        config.executable = Executable('/a/bogus/path')
        opt.version(config=config)
        self.assertIsNone(opt._version_cache[0])
        self.assertIsNone(opt._version_cache[1])

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
        self.assertEqual(parsed_output["iters"], 11)
        self.assertEqual(len(parsed_output["iteration_log"]), 12)
        self.assertEqual(parsed_output["incumbent_objective"], 7.0136459513364959e-25)
        self.assertIn("final_scaled_results", parsed_output.keys())
        self.assertIn(
            'IPOPT (w/o function evaluations)', parsed_output['cpu_seconds'].keys()
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
        self.assertEqual(parsed_output["iters"], 11)
        self.assertEqual(len(parsed_output["iteration_log"]), 12)
        self.assertEqual(parsed_output["incumbent_objective"], 7.0136459513364959e-25)
        self.assertIn("final_scaled_results", parsed_output.keys())
        self.assertIn('IPOPT', parsed_output['cpu_seconds'].keys())

    def test_empty_output_parsing(self):
        with self.assertLogs(
            "pyomo.contrib.solver.solvers.ipopt", level="WARNING"
        ) as logs:
            ipopt.Ipopt()._parse_ipopt_output(output=None)
        self.assertIn(
            "Returned output from ipopt was empty. Cannot parse for additional data.",
            logs.output[0],
        )

    def test_verify_ipopt_options(self):
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        opt._verify_ipopt_options(opt.config)
        self.assertEqual(opt.config.solver_options.value(), {'max_iter': 4})

        opt = ipopt.Ipopt(solver_options={'max_iter': 4}, time_limit=10)
        opt._verify_ipopt_options(opt.config)
        self.assertEqual(
            opt.config.solver_options.value(), {'max_iter': 4, 'max_cpu_time': 10}
        )

        # Finally, let's make sure it errors if someone tries to pass option_file_name
        opt = ipopt.Ipopt(
            solver_options={'max_iter': 4, 'option_file_name': 'myfile.opt'}
        )
        with self.assertRaisesRegex(
            ValueError,
            r'Pyomo generates the ipopt options file as part of the `solve` '
            r'method.  Add all options to ipopt.config.solver_options instead',
        ):
            opt._verify_ipopt_options(opt.config)

    def test_write_options_file(self):
        # If we have no options, nothing should happen (and no options
        # file should be added tot he set of options)
        opt = ipopt.Ipopt()
        opt._write_options_file('fakename', opt.config.solver_options)
        self.assertEqual(opt.config.solver_options.value(), {})
        # Pass it some options that ARE on the command line
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        opt._write_options_file('myfile', opt.config.solver_options)
        self.assertNotIn('option_file_name', opt.config.solver_options)
        self.assertFalse(os.path.isfile('myfile.opt'))
        # Now we are going to actually pass it some options that are NOT on
        # the command line
        opt = ipopt.Ipopt(solver_options={'custom_option': 4})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile.opt')
            opt._write_options_file(filename, opt.config.solver_options)
            self.assertIn('option_file_name', opt.config.solver_options)
            self.assertTrue(os.path.isfile(filename))
        # Make sure all options are writing to the file
        opt = ipopt.Ipopt(solver_options={'custom_option_1': 4, 'custom_option_2': 3})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile.opt')
            opt._write_options_file(filename, opt.config.solver_options)
            self.assertIn('option_file_name', opt.config.solver_options)
            self.assertTrue(os.path.isfile(filename))
            with open(filename, 'r') as f:
                data = f.readlines()
                self.assertEqual(
                    len(data) + 1, len(list(opt.config.solver_options.keys()))
                )

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

    def test_create_command_line(self):
        opt = ipopt.Ipopt()
        # No custom options, no file created. Plain and simple.
        result = opt._create_command_line('myfile', opt.config)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL'])
        # Custom command line options
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        result = opt._create_command_line('myfile', opt.config)
        self.assertEqual(
            result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4']
        )
        # Let's see if we correctly parse config.time_limit
        opt = ipopt.Ipopt(solver_options={'max_iter': 4}, time_limit=10)
        opt._verify_ipopt_options(opt.config)
        result = opt._create_command_line('myfile', opt.config)
        self.assertEqual(
            result,
            [
                str(opt.config.executable),
                'myfile.nl',
                '-AMPL',
                'max_iter=4',
                'max_cpu_time=10.0',
            ],
        )
        # Now let's do multiple command line options
        opt = ipopt.Ipopt(solver_options={'max_iter': 4, 'max_cpu_time': 10})
        opt._verify_ipopt_options(opt.config)
        result = opt._create_command_line('myfile', opt.config)
        self.assertEqual(
            result,
            [
                str(opt.config.executable),
                'myfile.nl',
                '-AMPL',
                'max_cpu_time=10',
                'max_iter=4',
            ],
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
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

    def test_ipopt_results(self):
        model = self.create_model()
        results = ipopt.Ipopt().solve(model)
        self.assertEqual(results.solver_name, 'ipopt')
        self.assertEqual(results.iteration_count, 11)
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

    def test_ipopt_options_file(self):
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

        # If the model name contains a quote, then the name needs
        # to be quoted
        model.name = "test'model'"
        results = ipopt.Ipopt().solve(
            model,
            solver_options={'bogus_option': 5},
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        self.assertEqual(results.termination_condition, TerminationCondition.error)
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertIn('OPTION_INVALID', results.solver_log)

        model.name = 'test"model'
        results = ipopt.Ipopt().solve(
            model,
            solver_options={'bogus_option': 5},
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        self.assertEqual(results.termination_condition, TerminationCondition.error)
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertIn('OPTION_INVALID', results.solver_log)

        # Because we are using universal=True for to_legal_filename,
        # using both single and double quotes will be OK
        model.name = 'test"\'model'
        results = ipopt.Ipopt().solve(
            model,
            solver_options={'bogus_option': 5},
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        self.assertEqual(results.termination_condition, TerminationCondition.error)
        self.assertEqual(results.solution_status, SolutionStatus.noSolution)
        self.assertIn('OPTION_INVALID', results.solver_log)

        if not is_windows:
            # This test is not valid on Windows, as {"} is not a valid
            # character in a directory name.
            with TempfileManager.new_context() as temp:
                dname = temp.mkdtemp()
                working_dir = os.path.join(dname, '"foo"')
                os.mkdir(working_dir)
                with self.assertRaisesRegex(ValueError, 'single and double'):
                    results = ipopt.Ipopt().solve(
                        model,
                        working_dir=working_dir,
                        solver_options={'bogus_option': 5},
                        raise_exception_on_nonoptimal_result=False,
                        load_solutions=False,
                    )


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
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
        print(LOG.getvalue())
        self.assertIn('OPTION_INVALID', LOG.getvalue())
        # Note: OF_ is stripped
        self.assertIn(
            'Read Option: "bogus_option". It is not a valid option', LOG.getvalue()
        )

        with self.assertRaisesRegex(ValueError, "unallowed ipopt option 'wantsol'"):
            results = ipopt.LegacyIpoptSolver().solve(
                model,
                tee=True,
                solver_options={'OF_wantsol': False},
                load_solutions=False,
            )
