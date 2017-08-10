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
from pyomo.environ import *
import sys, os, shutil
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

if __name__ == "__main__":
    unittest.main()
