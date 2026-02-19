# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os
import subprocess
import sys

import pyomo.environ as pyo

from pyomo.common import unittest, Executable
from pyomo.common.config import ConfigDict
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import ExecutableData
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.opt.base import SolverFactory
from pyomo.repn.plugins.gams_writer_v2 import GAMSWriter
from pyomo.contrib.solver.common.util import (
    NoDualsError,
    NoOptimalSolutionError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.common.results import TerminationCondition, SolutionStatus
import pyomo.contrib.solver.solvers.gams as gams

"""
Formatted after pyomo/pyomo/contrib/solver/test/solvers/test_ipopt.py
"""


gams_available = gams.GAMS().available()


@unittest.skipIf(not gams_available, "The 'gams' command is not available")
@unittest.pytest.mark.solver("gams")
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


@unittest.pytest.mark.solver("gams")
class TestGAMSSolutionLoader(unittest.TestCase):
    def test_get_reduced_costs_error(self):
        loader = gams.GMSSolutionLoader(None, None)
        with self.assertRaises(NoSolutionError):
            loader.get_primals()
        with self.assertRaises(NoDualsError):
            loader.get_duals()
        with self.assertRaises(NoReducedCostsError):
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

        # We are asserting if there is no solution, the SymbolMap for
        # variable length must be 0
        loader.get_primals()

        # if the model is infeasible, no dual information is returned
        with self.assertRaises(NoDualsError):
            loader.get_duals()


@unittest.skipIf(not gams_available, "The 'gams' command is not available")
@unittest.pytest.mark.solver("gams")
class TestGAMSInterface(unittest.TestCase):
    # _simple_model and _run_simple_model are standalone functions to
    # test gams execution
    def _simple_model(self, n):
        return """
            option limrow = 0;
            option limcol = 0;
            option solprint = off;
            set I / 1 * %s /;
            variables ans;
            positive variables x(I);
            equations obj;
            obj.. ans =g= sum(I, x(I));
            model test / all /;
            solve test using lp minimizing ans;
            """ % (n,)

    def _run_simple_model(self, config, n):
        solver_exec = config.executable.path()
        if solver_exec is None:
            return False
        with TempfileManager.new_context() as tempfile:
            tmpdir = tempfile.mkdtemp()
            test = os.path.join(tmpdir, 'test.gms')
            with open(test, 'w') as FILE:
                FILE.write(self._simple_model(n))
            result = subprocess.run(
                [solver_exec, test, "curdir=" + tmpdir, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return not result.returncode

    def test_class_member_list(self):
        opt = gams.GAMS()
        expected_list = [
            'CONFIG',
            'available',
            'config',
            'api_version',
            'is_persistent',
            'name',
            'solve',
            'version',
            # 'license_is_valid', # DEPRECATED
        ]
        method_list = [method for method in dir(opt) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_default_instantiation(self):
        opt = gams.GAMS()
        self.assertFalse(opt.is_persistent())
        self.assertTrue(opt.version())
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

    def test_available_cache(self):
        opt = gams.GAMS()
        self.assertTrue(
            opt.available() in {Availability.FullLicense, Availability.LimitedLicense}
        )
        # Now we will try with a custom config that has a fake path
        config = gams.GAMSConfig()
        config.executable = Executable('/a/bogus/path')
        opt = gams.GAMS()
        opt.config = config
        self.assertTrue(opt.available() == Availability.NotFound)

    def test_version(self):
        # GAMS should be available...
        opt = gams.GAMS()
        ver = opt.version()
        self.assertIsInstance(ver, tuple)
        self.assertEqual(len(ver), 3)
        self.assertIsInstance(ver[0], int)
        self.assertIsInstance(ver[1], int)
        self.assertIsInstance(ver[2], int)
        self.assertGreater(ver[0], 0)
        self.assertGreaterEqual(ver[1], 0)
        self.assertGreaterEqual(ver[2], 0)

        # Now we will try with a custom config that has a fake path
        config = gams.GAMSConfig()
        config.executable = Executable('/a/bogus/path')
        opt = gams.GAMS()
        opt.config = config
        self.assertIsNone(opt.version())

        # Now try pointing to "executables" that are not GAMS
        with TempfileManager as tmp:
            dname = tmp.mkdtemp()

            # Make sure we can run python files as if they we
            # executables (an issue on Windows)
            fname = os.path.join(dname, 'test.py')
            with open(fname, 'w') as F:
                F.write(f"#!{sys.executable}\nimport sysn\n")
            os.chmod(fname, 0o755)
            try:
                subprocess.run([fname])
            except OSError:
                raise unittest.SkipTest(
                    "python scripts are not registered as executable"
                )

            fname = os.path.join(dname, 'test_rc.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('HERE\\n')
sys.exit(2)
""")
            os.chmod(fname, 0o755)
            opt.config.executable = fname
            with LoggingIntercept() as LOG:
                self.assertIsNone(opt.version())
                self.assertEqual(opt.available(), Availability.NotFound)
            self.assertEqual(
                "Failed running GAMS command to get version (non-zero returncode 2):\n"
                "    FAIL\n    HERE\n",
                LOG.getvalue(),
            )

            fname = os.path.join(dname, 'test_bad_ver.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('GAMS Release : 26.1\\n')
sys.stdout.write('HERE\\n')
""")
            os.chmod(fname, 0o755)
            opt.config.executable = fname
            with LoggingIntercept() as LOG:
                self.assertIsNone(opt.version())
                self.assertEqual(opt.available(), Availability.NotFound)
            self.assertEqual(
                "Failed parsing GAMS version (version not found while parsing):\n"
                "    FAIL\n    GAMS Release : 26.1\n    HERE\n",
                LOG.getvalue(),
            )

            fname = os.path.join(dname, 'test_demo.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('*** GAMS Release     : 45.1.0 88bbff72\\n')
sys.stdout.write('*** GAMS Demo, for EULA and demo limitations\\n')
sys.stdout.write('HERE\\n')
""")
            os.chmod(fname, 0o755)
            opt.config.executable = fname
            with LoggingIntercept() as LOG:
                self.assertEqual(opt.version(), (45, 1, 0))
                self.assertEqual(opt.available(), Availability.LimitedLicense)
            self.assertEqual("", LOG.getvalue())

            fname = os.path.join(dname, 'test_full.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('*** GAMS Release     : 45.1.0 88bbff72\\n')
sys.stdout.write('*** Evaluation expiration date (GAMS base module)\\n')
sys.stdout.write('HERE\\n')
""")
            os.chmod(fname, 0o755)
            opt.config.executable = fname
            with LoggingIntercept() as LOG:
                self.assertEqual(opt.version(), (45, 1, 0))
                self.assertEqual(opt.available(), Availability.FullLicense)
            self.assertEqual("", LOG.getvalue())

    def test_write_gms_file(self):
        # We are creating a simple model with 1 variable to check for gams execution
        opt = gams.GAMS()
        config = gams.GAMSConfig()
        result = self._run_simple_model(config, 1)
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
                FILE.write(self._simple_model(1))
            result = subprocess.run(
                [solver_exec, filename, "curdir=" + dname, 'lo=0'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertTrue(result.returncode == 0)
            self.assertTrue(os.path.isfile(filename))


@unittest.skipIf(not gams_available, "The 'gams' command is not available")
@unittest.pytest.mark.solver("gams")
class TestGAMS(unittest.TestCase):
    def create_model(self):
        m = pyo.ConcreteModel('TestModel')
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))

        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        return m

    def create_infeasible_model(self):
        m = pyo.ConcreteModel('TestModel')
        m.x = pyo.Var(initialize=1.5, bounds=(-4, 4))
        m.y = pyo.Var(initialize=1.5, bounds=(-4, 4))

        m.c1 = pyo.Constraint(expr=m.x + m.y >= 5)
        m.c2 = pyo.Constraint(expr=-m.x + m.y >= 5)
        m.obj = pyo.Objective(rule=m.x + m.y, sense=pyo.minimize)
        return m

    def test_gams_config(self):
        # Test default initialization
        config = gams.GAMSConfig()
        self.assertTrue(config.load_solutions)
        self.assertIsInstance(config.solver_options, ConfigDict)
        self.assertIsInstance(config.executable, ExecutableData)

        # Test custom initialization
        solver = SolverFactory('gams_v2', executable='/path/to/exe')
        self.assertFalse(solver.config.tee)
        self.assertIsNone(solver.config.executable.path())
        self.assertTrue(solver.config.executable._registered_name.startswith('/path'))

    @unittest.skipIf(not gams.gdxcc_available, "'gdx' requires the gdx/gdxcc module")
    def test_gams_solve_gdx(self):
        # Gut check - does it solve?
        model = self.create_model()
        result = gams.GAMS().solve(model, writer_config={'put_results_format': 'gdx'})
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, -1004)
        self.assertAlmostEqual(model.x.value, 5)
        self.assertAlmostEqual(model.y.value, -5)

        model.obj.sense = pyo.maximize
        result = gams.GAMS().solve(model, writer_config={'put_results_format': 'gdx'})
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, 1006)
        self.assertAlmostEqual(model.x.value, -5)
        self.assertAlmostEqual(model.y.value, 5)

    def test_gams_solve_dat(self):
        # Gut check - does it solve?
        model = self.create_model()
        result = gams.GAMS().solve(model, writer_config={'put_results_format': 'dat'})
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, -1004)
        self.assertAlmostEqual(model.x.value, 5)
        self.assertAlmostEqual(model.y.value, -5)

        model.obj.sense = pyo.maximize
        result = gams.GAMS().solve(model, writer_config={'put_results_format': 'dat'})
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, 1006)
        self.assertAlmostEqual(model.x.value, -5)
        self.assertAlmostEqual(model.y.value, 5)

    @unittest.skipIf(not gams.gdxcc_available, "'gdx' requires the gdx/gdxcc module")
    def test_gams_solve_noload_gdx(self):
        # Gut check - does it solve?
        model = self.create_model()
        result = gams.GAMS().solve(
            model, load_solutions=False, writer_config={'put_results_format': 'gdx'}
        )
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, -1004)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

        model.obj.sense = pyo.maximize
        result = gams.GAMS().solve(
            model, load_solutions=False, writer_config={'put_results_format': 'gdx'}
        )
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, 1006)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

    def test_gams_solve_noload_dat(self):
        # Gut check - does it solve?
        model = self.create_model()
        result = gams.GAMS().solve(
            model, load_solutions=False, writer_config={'put_results_format': 'dat'}
        )
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, -1004)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

        model.obj.sense = pyo.maximize
        result = gams.GAMS().solve(
            model, load_solutions=False, writer_config={'put_results_format': 'dat'}
        )
        self.assertEqual(
            result.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(result.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(result.incumbent_objective, 1006)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

    @unittest.skipIf(not gams.gdxcc_available, "'gdx' requires the gdx/gdxcc module")
    def test_gams_solve_infeasible_gdx(self):
        # Gut check - does it solve?
        model = self.create_infeasible_model()
        result = gams.GAMS().solve(
            model,
            raise_exception_on_nonoptimal_result=False,
            writer_config={'put_results_format': 'gdx'},
        )
        self.assertEqual(
            result.termination_condition, TerminationCondition.provenInfeasible
        )
        self.assertEqual(result.solution_status, SolutionStatus.infeasible)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

        model.obj.sense = pyo.maximize
        with self.assertRaisesRegex(
            NoOptimalSolutionError, "Solver did not find the optimal solution."
        ):
            result = gams.GAMS().solve(
                model, writer_config={'put_results_format': 'gdx'}
            )
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

    def test_gams_solve_infeasible_dat(self):
        # Gut check - does it solve?
        model = self.create_infeasible_model()
        result = gams.GAMS().solve(
            model,
            raise_exception_on_nonoptimal_result=False,
            writer_config={'put_results_format': 'dat'},
        )
        self.assertEqual(
            result.termination_condition, TerminationCondition.provenInfeasible
        )
        self.assertEqual(result.solution_status, SolutionStatus.infeasible)
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

        model.obj.sense = pyo.maximize
        with self.assertRaisesRegex(
            NoOptimalSolutionError, "Solver did not find the optimal solution."
        ):
            result = gams.GAMS().solve(
                model, writer_config={'put_results_format': 'dat'}
            )
        self.assertAlmostEqual(model.x.value, 1.5)
        self.assertAlmostEqual(model.y.value, 1.5)

    def test_bad_gams_executable(self):
        model = self.create_model()
        solver = gams.GAMS()
        solver.config.executable = "/path/to/bogus/gams"

        with self.assertRaisesRegex(
            ApplicationError, "GAMS: 'gams' executable not found"
        ):
            solver.solve(model)

        # Now try pointing to "executables" that are not GAMS
        with TempfileManager as tmp:
            dname = tmp.mkdtemp()
            solver.config.working_dir = dname

            # Make sure we can run python files as if they we
            # executables (an issue on Windows)
            fname = os.path.join(dname, 'test.py')
            with open(fname, 'w') as F:
                F.write(f"#!{sys.executable}\nimport sys\n")
            try:
                subprocess.run([fname])
            except OSError:
                raise unittest.SkipTest(
                    "python scripts are not registered as executable"
                )

            fname = os.path.join(dname, 'test_rc.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
lst = sys.argv[1].replace('.gms', '.lst')
for arg in sys.argv:
    if arg.startswith('o='):
        lst = arg[2:]
with open(lst, 'w') as LST:
    LST.write("Here is a multiline\\nlisting file.\\n")
sys.stderr.write('FAIL\\n')
sys.stdout.write('HERE\\n')
sys.exit(2)
""")
            os.chmod(fname, 0o755)
            solver.config.executable = fname
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"GAMS process encountered an error \(returncode=2\).\n"
                    r"Check listing file for details.",
                ):
                    solver.solve(model)
            self.assertEqual(
                "\n".join(
                    [
                        "GAMS process encountered an error (returncode=2).",
                        "Check listing file for details.",
                        "",
                        "FAIL",
                        "HERE",
                        "",
                        "GAMS Listing file:",
                        "",
                        "Here is a multiline",
                        "listing file.",
                    ]
                )
                + "\n",
                LOG.getvalue(),
            )

        with TempfileManager as tmp:
            dname = tmp.mkdtemp()
            solver.config.working_dir = dname

            fname = os.path.join(dname, 'test_rc_no_lst.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('HERE\\n')
sys.exit(2)
""")
            os.chmod(fname, 0o755)
            solver.config.executable = fname
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"GAMS process encountered an error \(returncode=2\).\n"
                    r"Check listing file for details.",
                ):
                    solver.solve(model)
            self.assertEqual(
                "\n".join(
                    [
                        "GAMS process encountered an error (returncode=2).",
                        "Check listing file for details.",
                        "",
                        "FAIL",
                        "HERE",
                    ]
                )
                + "\n",
                LOG.getvalue(),
            )

        with TempfileManager as tmp:
            dname = tmp.mkdtemp()
            solver.config.working_dir = dname

            fname = os.path.join(dname, 'test_rc3_no_lst.py')
            with open(fname, 'w') as F:
                F.write(f"""\
#!{sys.executable}
import sys
sys.stderr.write('FAIL\\n')
sys.stdout.write('HERE\\n')
sys.exit(3)
""")
            os.chmod(fname, 0o755)
            solver.config.executable = fname
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"GAMS process encountered an error \(returncode=3\).\n"
                    r"Error rc=3 \(GAMS execution error\), to be determined later.\n"
                    r"Check listing file for details.",
                ):
                    solver.solve(model)
            self.assertEqual(
                "\n".join(
                    [
                        "GAMS process encountered an error (returncode=3).",
                        "Error rc=3 (GAMS execution error), to be determined later.",
                        "Check listing file for details.",
                        "",
                        "FAIL",
                        "HERE",
                    ]
                )
                + "\n",
                LOG.getvalue(),
            )
