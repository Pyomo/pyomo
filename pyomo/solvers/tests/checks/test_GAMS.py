#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import pyutilib.th as unittest
import pyutilib.subprocess
from pyutilib.misc import setup_redirect, reset_redirect
from pyomo.environ import *
from six import StringIO
import contextlib, sys, os, shutil
from tempfile import mkdtemp
import subprocess

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
                                                         'results.dat')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir,
                                                         'resultsstat.dat')))

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
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         'results.dat')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir,
                                                         'resultsstat.dat')))

            shutil.rmtree(tmpdir)

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

    def tearDown(self):
        """Clean up temporary directory after tests are over."""
        shutil.rmtree(self.tmpdir)

    def _check_logfile(self, exists=True, expected=None):
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
        self.assertTrue(logfile_contents)
        if expected:
            self.assertEqual(logfile_contents, expected)


@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
class GAMSLogfileGmsTests(GAMSLogfileTestBase):
    """Test class for testing permultations of tee and logfile options.

    The tests build a simple model and solve it using the different options
    using the gams command directly.

    """

    def test_no_tee(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with redirected_subprocess_call() as output:
                opt.solve(self.m, tee=False)
        self.assertFalse(output.getvalue())
        self._check_logfile(exists=False)

    def test_tee(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with redirected_subprocess_call() as output:
                opt.solve(self.m, tee=True)
        self.assertTrue(output.getvalue())
        self._check_logfile(exists=False)

    def test_logfile(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with redirected_subprocess_call() as output:
                opt.solve(self.m, logfile=self.logfile)
        self.assertFalse(output.getvalue())
        self._check_logfile(exists=True)

    def test_tee_and_logfile(self):
        with SolverFactory("gams", solver_io="gms") as opt:
            with redirected_subprocess_call() as output:
                opt.solve(self.m, logfile=self.logfile, tee=True)
        self.assertTrue(output.getvalue())
        self._check_logfile(exists=True, expected=output.getvalue())


@unittest.skipIf(not gamspy_available, "The 'gams' python bindings are not available")
class GAMSLogfilePyTests(GAMSLogfileTestBase):
    """Test class for testing permultations of tee and logfile options.

    The tests build a simple model and solve it using the different options
    using the python gams bindings.

    """

    def test_no_tee(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with redirected_stdout() as output:
                opt.solve(self.m, tee=False)
        self.assertFalse(output.getvalue())
        self._check_logfile(exists=False)

    def test_tee(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with redirected_stdout() as output:
                opt.solve(self.m, tee=True)
        self.assertTrue(output.getvalue())
        self._check_logfile(exists=False)

    def test_logfile(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with redirected_stdout() as output:
                opt.solve(self.m, logfile=self.logfile)
        self._check_logfile(exists=True)
        self.assertFalse(output.getvalue())

    def test_tee_and_logfile(self):
        with SolverFactory("gams", solver_io="python") as opt:
            with redirected_stdout() as output:
                opt.solve(self.m, logfile=self.logfile, tee=True)
        self.assertTrue(output.getvalue())
        self._check_logfile(exists=True, expected=output.getvalue())


@contextlib.contextmanager
def redirected_stdout():
    """Temporarily redirect stdout into a string buffer."""
    output = StringIO()
    try:
        setup_redirect(output)
        yield output
    finally:
        reset_redirect()


@contextlib.contextmanager
def redirected_subprocess_call():
    """Temporarily redirect subprocess calls stdout into a string buffer."""
    output = StringIO()
    old_call = subprocess.call

    def call(*args, **kwargs):
        returncode, out = pyutilib.subprocess.run(*args, **kwargs)
        output.write(out)
        return returncode

    try:
        subprocess.call = call
        yield output
    finally:
        subprocess.call = old_call


if __name__ == "__main__":
    unittest.main()
