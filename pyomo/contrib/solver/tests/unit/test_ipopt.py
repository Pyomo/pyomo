#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt


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
        with self.assertRaises(RuntimeError):
            loader.get_reduced_costs()

        # Set _nl_info to something completely bogus but is not None
        class NLInfo:
            pass

        loader._nl_info = NLInfo()
        loader._nl_info.eliminated_vars = [1, 2, 3]
        with self.assertRaises(NotImplementedError):
            loader.get_reduced_costs()
        # Reset _nl_info so we can ensure we get an error
        # when _sol_data is None
        loader._nl_info.eliminated_vars = []
        with self.assertRaises(DeveloperError):
            loader.get_reduced_costs()


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestIpoptInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = ipopt.Ipopt()
        expected_list = [
            'Availability',
            'CONFIG',
            'config',
            'available',
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
