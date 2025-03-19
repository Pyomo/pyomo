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
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.common.errors import DeveloperError
import pyomo.contrib.solver.solvers.ipopt as ipopt
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.common import unittest, Executable
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter


"""
TODO:
    - Test unique configuration options
    - Test unique results options
    - Ensure that `*.opt` file is only created when needed
    - Ensure options are correctly parsing to env or opt file
    - Failures at appropriate times
"""


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

    def test_write_options_file(self):
        # If we have no options, we should get false back
        opt = ipopt.Ipopt()
        result = opt._write_options_file('fakename', None)
        self.assertFalse(result)
        # Pass it some options that ARE on the command line
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        result = opt._write_options_file('myfile', opt.config.solver_options)
        self.assertFalse(result)
        self.assertFalse(os.path.isfile('myfile.opt'))
        # Now we are going to actually pass it some options that are NOT on
        # the command line
        opt = ipopt.Ipopt(solver_options={'custom_option': 4})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile')
            result = opt._write_options_file(filename, opt.config.solver_options)
            self.assertTrue(result)
            self.assertTrue(os.path.isfile(filename + '.opt'))
        # Make sure all options are writing to the file
        opt = ipopt.Ipopt(solver_options={'custom_option_1': 4, 'custom_option_2': 3})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'myfile')
            result = opt._write_options_file(filename, opt.config.solver_options)
            self.assertTrue(result)
            self.assertTrue(os.path.isfile(filename + '.opt'))
            with open(filename + '.opt', 'r') as f:
                data = f.readlines()
                self.assertEqual(len(data), len(list(opt.config.solver_options.keys())))

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
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL'])
        # Custom command line options
        opt = ipopt.Ipopt(solver_options={'max_iter': 4})
        result = opt._create_command_line('myfile', opt.config, False)
        self.assertEqual(
            result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4']
        )
        # Let's see if we correctly parse config.time_limit
        opt = ipopt.Ipopt(solver_options={'max_iter': 4}, time_limit=10)
        result = opt._create_command_line('myfile', opt.config, False)
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
        result = opt._create_command_line('myfile', opt.config, False)
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
        # Let's now include if we "have" an options file
        result = opt._create_command_line('myfile', opt.config, True)
        self.assertEqual(
            result,
            [
                str(opt.config.executable),
                'myfile.nl',
                '-AMPL',
                'option_file_name=myfile.opt',
                'max_cpu_time=10',
                'max_iter=4',
            ],
        )
        # Finally, let's make sure it errors if someone tries to pass option_file_name
        opt = ipopt.Ipopt(
            solver_options={'max_iter': 4, 'option_file_name': 'myfile.opt'}
        )
        with self.assertRaises(ValueError):
            result = opt._create_command_line('myfile', opt.config, False)


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
        self.assertTrue(solver.config.executable.startswith('/path'))

        # Change value on a solve call
        # model = self.create_model()
        # result = solver.solve(model, tee=True)
