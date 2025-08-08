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

from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
import pyomo.environ as pyo
from pyomo.common.fileutils import ExecutableData
from pyomo.common.config import ConfigDict
from pyomo.common.errors import DeveloperError
import pyomo.contrib.solver.solvers.gams as gams
from pyomo.contrib.solver.common.util import NoSolutionError
from pyomo.opt.base import SolverFactory
from pyomo.common import unittest, Executable
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.gams_writer_v2 import GAMSWriter
import pdb

"""
Formatted after pyomo/pyomo/contrib/solver/test/solvers/test_ipopt.py
"""


gams_available = gams.GAMS().available()


@unittest.skipIf(not gams_available, "The 'gams' command is not available")
class TestGAMSSolverConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = gams.GAMSConfig()
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
        self.assertIsInstance(config.writer_config, type(GAMSWriter.CONFIG()))

    def test_custom_instantiation(self):
        config = gams.GAMSConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertIsNone(config.time_limit)
        # Default should be `gams`
        self.assertIsNotNone(str(config.executable))
        self.assertIn('gams', str(config.executable))
        # Set to a totally bogus path
        config.executable = Executable('/bogus/path')
        self.assertIsNone(config.executable.executable)
        self.assertFalse(config.executable.available())


class TestGAMSSolutionLoader(unittest.TestCase):
    def test_get_reduced_costs_error(self):
        loader = gams.GMSSolutionLoader(None, None)
        with self.assertRaises(RuntimeError):
            loader.get_primals()
            loader.get_duals()
            loader.get_reduced_costs()

        # Set _gms_info to something completely bogus but is not None
        # Set the var_symbol_map and con_symbol_map to empty SymbolMap object type
        class GAMSInfo:
            pass

        class GDXData:
            pass

        loader._gms_info = GAMSInfo()
        loader._gms_info.var_symbol_map = SymbolMap(NumericLabeler('x'))
        loader._gms_info.con_symbol_map = SymbolMap(NumericLabeler('c'))

        # We are asserting if there is no solution, the SymbolMap for variable length must be 0
        loader.get_primals()

        # if the model is infeasible, no dual information is returned
        with self.assertRaises(RuntimeError):
            loader.get_duals()


@unittest.skipIf(not gams_available, "The 'gams' command is not available")
class TestGAMSInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = gams.GAMS()
        expected_list = [
            'CONFIG',
            'available',
            'config',
            'is_persistent',
            'name',
            'solve',
            'version',
        ]
        method_list = [method for method in dir(opt) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = gams.GAMS()
        self.assertFalse(opt.is_persistent())
        self.assertIsNotNone(opt.version())
        self.assertEqual(opt.name, 'gams')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())

    def test_context_manager(self):
        with gams.GAMS() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertIsNotNone(opt.version())
            self.assertEqual(opt.name, 'gams')
            self.assertEqual(opt.CONFIG, opt.config)
            self.assertTrue(opt.available())

    def test_available(self):
        opt = gams.GAMS()
        self.assertTrue(opt.available())
        # Now we will try with a custom config that has a fake path
        config = gams.GAMSConfig()
        config.executable = Executable('/a/bogus/path')
        with self.assertRaises(NameError):
            opt.available(config=config)

        # _run_simple_model will return False because of the invalid path
        self.assertFalse(opt._run_simple_model(config, 1))

    def test_version(self):
        opt = gams.GAMS()
        self.assertIsNotNone(opt.version())

    def test_write_gms_file(self):
        # We are creating a simple model with 1 variable to check for gams execution
        opt = gams.GAMS()
        config = gams.GAMSConfig()
        result = opt._run_simple_model(config, 1)
        self.assertTrue(result)

        # Pass it some options that ARE on the command line and create a .gms file
        # Currently solver_options is not implemented in the new interface
        solver_exec = config.executable.path()
        opt = gams.GAMS(solver_options={'iterLim': 1})
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            if not os.path.exists(dname):
                os.mkdir(dname)
            filename = os.path.join(dname, 'test.gms')
            with open(filename, 'w') as FILE:
                FILE.write(opt._simple_model(1))
            result = subprocess.run(
                [solver_exec, filename, "curdir=" + dname, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertTrue(result.returncode == 0)
            self.assertTrue(os.path.isfile(filename))


class TestGAMS(unittest.TestCase):
    def create_model(self):
        model = pyo.ConcreteModel('TestModel')
        model.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        model.y = pyo.Var(initialize=1.5, bounds=(-5, 5))

        def dummy_equation(m):
            return (1.0 - m.x) + 100.0 * (m.y - m.x)

        model.obj = pyo.Objective(rule=dummy_equation, sense=pyo.minimize)
        return model

    def test_gams_config(self):
        # Test default initialization
        config = gams.GAMSConfig()
        self.assertTrue(config.load_solutions)
        self.assertIsInstance(config.solver_options, ConfigDict)
        self.assertIsInstance(config.executable, ExecutableData)

        # Test custom initialization
        solver = SolverFactory('gams_v2', executable='/path/to/exe')
        self.assertFalse(solver.config.tee)
        self.assertTrue(solver.config.executable.startswith('/path'))

        # Change value on a solve call
        # config = gams.GAMSConfig()
        with TempfileManager.new_context() as temp:
            dname = temp.mkdtemp()
            # if working_dir is not specified, the tmpdir is deleted before solve can happen
            solver = SolverFactory('gams_v2', working_dir=dname)
            model = self.create_model()
            solver.solve(model, tee=False, load_solutions=False)
