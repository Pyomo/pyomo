#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.environ import ConcreteModel, Var, Objective, Constraint, maximize, Expression, log10
from pyomo.opt import SolverFactory, TerminationCondition

from pyomo.solvers.plugins.solvers.GAMS import (
    GAMSShell, GAMSDirect, gdxcc_available
)
import pyutilib.th as unittest
from pyutilib.misc import capture_output
import os, shutil
from tempfile import mkdtemp

opt_py = SolverFactory('gams', solver_io='python')
gamspy_available = opt_py.available(exception_flag=False)
if gamspy_available:
    from gams.workspace import GamsExceptionExecution

opt_gms = SolverFactory('gams', solver_io='gms')
gamsgms_available = opt_gms.available(exception_flag=False)


class GAMSTests(unittest.TestCase):

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_check_expr_eval_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.e = Expression(expr= log10(m.x) + 5)
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.e)

            self.assertRaises(GamsExceptionExecution, opt.solve, m)

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_check_expr_eval_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.e = Expression(expr= log10(m.x) + 5)
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.e)

            self.assertRaises(ValueError, opt.solve, m)

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_file_removal_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            tmpdir = mkdtemp()

            results = opt.solve(m, tmpdir=tmpdir)

            self.assertTrue(os.path.exists(tmpdir))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gjo0.gms')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gjo0.lst')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gdb0.gdx')))

            os.rmdir(tmpdir)

            results = opt.solve(m, tmpdir=tmpdir)

            self.assertFalse(os.path.exists(tmpdir))

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_file_removal_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            tmpdir = mkdtemp()

            results = opt.solve(m, tmpdir=tmpdir)

            self.assertTrue(os.path.exists(tmpdir))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'model.gms')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'output.lst')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'GAMS_MODEL_p.gdx')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'GAMS_MODEL_s.gdx')))

            os.rmdir(tmpdir)

            results = opt.solve(m, tmpdir=tmpdir)

            self.assertFalse(os.path.exists(tmpdir))

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_keepfiles_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            tmpdir = mkdtemp()

            results = opt.solve(m, tmpdir=tmpdir, keepfiles=True)

            self.assertTrue(os.path.exists(tmpdir))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gjo0.gms')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gjo0.lst')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gdb0.gdx')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         '_gams_py_gjo0.pf')))

            shutil.rmtree(tmpdir)

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_keepfiles_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            tmpdir = mkdtemp()

            results = opt.solve(m, tmpdir=tmpdir, keepfiles=True)

            self.assertTrue(os.path.exists(tmpdir))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         'model.gms')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         'output.lst')))
            if gdxcc_available:
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir, 'GAMS_MODEL_p.gdx')))
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir, 'results_s.gdx')))
            else:
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir, 'results.dat')))
                self.assertTrue(os.path.exists(os.path.join(
                    tmpdir, 'resultsstat.dat')))

            shutil.rmtree(tmpdir)

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_fixed_var_sign_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.y = Var()
            m.z = Var()
            m.z.fix(-3)
            m.c1 = Constraint(expr= m.x + m.y - m.z == 0)
            m.c2 = Constraint(expr= m.z + m.y - m.z >= -10000)
            m.c3 = Constraint(expr= -3 * m.z + m.y - m.z >= -10000)
            m.c4 = Constraint(expr= -m.z + m.y - m.z >= -10000)
            m.c5 = Constraint(expr= m.x <= 100)
            m.o = Objective(expr= m.x, sense=maximize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_fixed_var_sign_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.y = Var()
            m.z = Var()
            m.z.fix(-3)
            m.c1 = Constraint(expr= m.x + m.y - m.z == 0)
            m.c2 = Constraint(expr= m.z + m.y - m.z >= -10000)
            m.c3 = Constraint(expr= -3 * m.z + m.y - m.z >= -10000)
            m.c4 = Constraint(expr= -m.z + m.y - m.z >= -10000)
            m.c5 = Constraint(expr= m.x <= 100)
            m.o = Objective(expr= m.x, sense=maximize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_long_var_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            x = m.a23456789012345678901234567890123456789012345678901234567890123 = Var()
            y = m.b234567890123456789012345678901234567890123456789012345678901234 = Var()
            z = m.c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 = Var()
            w = m.d01234567890 = Var()
            m.c1 = Constraint(expr= x + y + z + w == 0)
            m.c2 = Constraint(expr= x >= 10)
            m.o = Objective(expr= x)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_long_var_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            x = m.a23456789012345678901234567890123456789012345678901234567890123 = Var()
            y = m.b234567890123456789012345678901234567890123456789012345678901234 = Var()
            z = m.c23456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 = Var()
            w = m.d01234567890 = Var()
            m.c1 = Constraint(expr= x + y + z + w == 0)
            m.c2 = Constraint(expr= x >= 10)
            m.o = Objective(expr= x)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_subsolver_notation(self):
        opt1 = SolverFactory("gams:ipopt", solver_io="gms")
        self.assertTrue(isinstance(opt1, GAMSShell))
        self.assertEqual(opt1.options["solver"], "ipopt")

        opt2 = SolverFactory("gams:baron", solver_io="python")
        self.assertTrue(isinstance(opt2, GAMSDirect))
        self.assertEqual(opt2.options["solver"], "baron")

        opt3 = SolverFactory("py:gams")
        self.assertTrue(isinstance(opt3, GAMSDirect))
        opt3.options["keepfiles"] = True
        self.assertEqual(opt3.options["keepfiles"], True)

        opt4 = SolverFactory("py:gams:cbc")
        self.assertTrue(isinstance(opt4, GAMSDirect))
        self.assertEqual(opt4.options["solver"], "cbc")

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_options_py(self):
        with SolverFactory("gams", solver_io="python") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            opt.options["load_solutions"] = False # set option
            opt.solve(m) # use option
            self.assertEqual(m.x.value, None)
            opt.solve(m, load_solutions=True) # overwrite option
            self.assertEqual(m.x.value, 10)

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_options_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr= m.x >= 10)
            m.o = Objective(expr= m.x)

            opt.options["load_solutions"] = False # set option
            opt.solve(m) # use option
            self.assertEqual(m.x.value, None)
            opt.solve(m, load_solutions=True) # overwrite option
            self.assertEqual(m.x.value, 10)

    @unittest.skipIf(not gamspy_available,
                     "The 'gams' python bindings are not available")
    def test_version_py(self):
        with SolverFactory("gams", solver_io="python") as opt:
            self.assertIsNotNone(opt.version())

    @unittest.skipIf(not gamsgms_available,
                     "The 'gams' executable is not available")
    def test_version_gms(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            self.assertIsNotNone(opt.version())

class GAMSLogfileTestBase(unittest.TestCase):
    def setUp(self):
        """Set up model and temporary directory."""
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr= m.x >= 10)
        m.o = Objective(expr= m.x)
        self.m = m
        self.tmpdir = mkdtemp()
        self.logfile = os.path.join(self.tmpdir, 'logfile.log')
        self.characteristic_output_string = "Starting compilation"

    def tearDown(self):
        """Clean up temporary directory after tests are over."""
        shutil.rmtree(self.tmpdir)

    def _check_logfile(self, exists=True):
        """Check for logfiles existence and contents.

        exists=True:
            Whether to check if the logfile exists or doesn't exist.
        expected=None:
            Optionally check that the logfiles contents is equal to this value.

        """
        if not exists:
            self.assertFalse(os.path.exists(self.logfile))
            return

        self.assertTrue(os.path.exists(self.logfile))
        with open(self.logfile) as f:
            logfile_contents = f.read()
        self.assertIn(self.characteristic_output_string, logfile_contents)

    def _check_stdout(self, output_string, exists=True):
        if exists:
            # Starting Compilation is outputted by the solver itself which in this
            # case should be printed to stdout and captured
            self.assertIn(self.characteristic_output_string, output_string)
        else:
            # Note that we do not check that the output string is completely
            # empty as certain platforms (like linux) output other lines to
            # stdout where as windows for example does not. We instead just
            # check that the characteristic string is missing.
            self.assertNotIn(self.characteristic_output_string, output_string)


@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
class GAMSLogfileGmsTests(GAMSLogfileTestBase):
    """Test class for testing permultations of tee and logfile options.

    The tests build a simple model and solve it using the different options
    using the gams command directly.

    """

    def test_no_tee(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=False)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=False)

    def test_tee(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=False)

    def test_logfile(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=True)

    def test_tee_and_logfile(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=True)


@unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
class GAMSLogfilePyTests(GAMSLogfileTestBase):
    """Test class for testing permultations of tee and logfile options.

    The tests build a simple model and solve it using the different options
    using the python gams bindings.

    """

    def test_no_tee(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=False)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=False)

    def test_tee(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with capture_output() as output:
                opt.solve(self.m, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=False)

    def test_logfile(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile)
        self._check_stdout(output.getvalue(), exists=False)
        self._check_logfile(exists=True)

    def test_tee_and_logfile(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with capture_output() as output:
                opt.solve(self.m, logfile=self.logfile, tee=True)
        self._check_stdout(output.getvalue(), exists=True)
        self._check_logfile(exists=True)



if __name__ == "__main__":
    unittest.main()
