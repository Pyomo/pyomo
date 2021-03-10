#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for pyomo.opt.blackbox.problem
#

import os
from os.path import abspath, dirname
pyomodir = dirname(dirname(dirname(dirname(abspath(__file__)))))
pyomodir += os.sep
currdir = dirname(abspath(__file__))+os.sep

from filecmp import cmp
import xml.etree.ElementTree as ET
import pyomo.common.unittest as unittest

import pyomo.opt
import pyomo.opt.blackbox
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output

old_tempdir = TempfileManager.tempdir

class TestProblem1(pyomo.opt.blackbox.MixedIntOptProblem):

    def __init__(self):
        pyomo.opt.blackbox.MixedIntOptProblem.__init__(self)
        self.real_lower=[0.0, -1.0, 1.0, None]
        self.real_upper=[None, 0.0, 2.0, -1.0]
        self.nreal=4

    def function_value(self, point):
        self.validate(point)
        return point.reals[0] - point.reals[1] + (point.reals[2]-1.5)**2 + (point.reals[3]+2)**4


class TestProblem2(pyomo.opt.blackbox.MixedIntOptProblem):

    def __init__(self):
        pyomo.opt.blackbox.MixedIntOptProblem.__init__(self)
        self.real_lower=[0.0]
        self.real_upper=[1.0]
        self.int_lower=[0.0]
        self.int_upper=[1.0]
        self.nreal=1
        self.nint=1
        self.nbinary=2

    def function_value(self, point):
        self.validate(point)
        return point.reals[0] + point.ints[0] + point.bits[0] + point.bits[1]


class RealProblem3(pyomo.opt.blackbox.RealOptProblem):

    def __init__(self):
        pyomo.opt.blackbox.RealOptProblem.__init__(self)
        self.lower=[-100.0]*4
        self.upper=[ 100.0]*4
        self.nvars=4
        self.ncons=4
        self.response_types = [pyomo.opt.blackbox.response_enum.FunctionValue,
                                pyomo.opt.blackbox.response_enum.FunctionValues,
                                pyomo.opt.blackbox.response_enum.Gradient,
                                pyomo.opt.blackbox.response_enum.Hessian,
                                pyomo.opt.blackbox.response_enum.NonlinearConstraintValues,
                                pyomo.opt.blackbox.response_enum.Jacobian]

    def function_value(self, point):
        return point.vars[0] - point.vars[1] + (point.vars[2]-1.5)**2 + (point.vars[3]+2)**4

    def gradient(self, point):
        return [1, -1, 2*(point.vars[2]-1.5), 4*(point.vars[3]+2)**3]

    def hessian(self, point):
        H = []
        H.append( (2,2,2) )
        H.append( (3,3,12*(point.vars[3]+2)**2) )
        return H

    def nonlinear_constraint_values(self, point):
        C = []
        C.append( sum(point.vars) )
        C.append( sum(x**2 for x in point.vars) )
        return C

    def jacobian(self, point):
        J = []
        for j in range(self.nvars):
            J.append( (0,j,1) )
        for j in range(self.nvars):
            J.append( (1,j,2*point.vars[j]) )
        return J



class TestDakotaMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        self.do_setup(False)
        TempfileManager.tempdir = currdir

    def do_setup(self,flag):
        TempfileManager.tempdir = currdir
        self.problem=TestProblem1()
        self.rproblem=RealProblem3()

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test_main(self):
        self.problem.main(['test_main', currdir+'request1.din', currdir+'results1.out'], format='dakota')
        with open(currdir+'results1.out', 'r') as f1, \
            open(currdir+'results1.dout', 'r') as f2:
                result = float(f1.read().strip().replace(' ASV_1', ''))
                baseline = float(f2.read().strip().replace(' ASV_1', ''))
        self.assertAlmostEqual(result, baseline)

class TestColinMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        self.do_setup(False)
        TempfileManager.tempdir = currdir

    def do_setup(self,flag):
        TempfileManager.tempdir = currdir
        self.ps = pyomo.opt.SolverFactory('ps')
        self.problem=TestProblem1()
        self.rproblem=RealProblem3()

    def compare_xml(self, file1, file2):
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            f1_contents = f1.read().replace('>', '> ').replace('</', ' </').split()
            f2_contents = f2.read().replace('>', '> ').replace('</', ' </').split()
            for item1, item2 in zip(f1_contents, f2_contents):
                try:
                    self.assertAlmostEqual(float(item1), float(item2))
                except:
                    self.assertEqual(item1, item2)

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test_main(self):
        self.problem.main(['test_main', currdir+'request1.xml', currdir+'results1.out'])
        self.compare_xml(currdir+'results1.out', currdir+'results1.xml')

    def test_main_a(self):
        self.problem.main(['test_main', currdir+'request1a.xml', currdir+'results1a.out'])
        self.compare_xml(currdir+'results1a.out', currdir+'results1.xml')

    def Xtest_rmain(self):
        self.rproblem.main(['test_main', currdir+'request3.xml', currdir+'results3.out'])
        self.assertTrue(cmp(currdir+'results3.out', currdir+'results3.xml'))

    def test_main_2(self):
        self.problem=TestProblem2()
        self.problem.main(['test_main', currdir+'request4.xml', currdir+'results4.out'])
        self.compare_xml(currdir+'results4.out', currdir+'results4.xml')

    def test_error2(self):
        try:
            self.problem.main(['test_main', currdir+'request1.xml', currdir+'results1.out'], 'foo')
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def Xtest_error4(self):
        self.problem.main(['test_main', currdir+'request2.xml', currdir+'results2.out'])
        self.compare_xml(currdir+'results2.out', currdir+'results2.xml')


class TestOptProblem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        self.do_setup(False)
        TempfileManager.tempdir = currdir

    def do_setup(self,flag):
        TempfileManager.tempdir = currdir
        self.ps = pyomo.opt.SolverFactory('ps')
        self.problem=TestProblem1()
        self.rproblem=RealProblem3()

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        TempfileManager.tempdir = old_tempdir

    def test_error1(self):
        point = pyomo.opt.blackbox.MixedIntVars()
        point.reals = [1.0]
        try:
            self.problem.validate(point)
            self.fail("Expected ValueError")
        except ValueError:
            pass
        point.reals = [1.0] * 4
        point.ints = [1.0] * 4
        try:
            self.problem.validate(point)
            self.fail("Expected ValueError")
        except ValueError:
            pass
        point.ints = []
        point.bits = [1]
        try:
            self.problem.validate(point)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_error3(self):
        try:
            self.problem.main(['test_main', currdir+'request0.xml', currdir+'results0.out'])
            self.fail("Expected IOError")
        except IOError:
            pass

    def test_error5(self):
        self.problem=TestProblem2()
        point = pyomo.opt.blackbox.MixedIntVars()
        point.reals = [1.0]
        point.ints = [1]
        point.bits = [0, 1]
        self.problem.validate(point)
        try:
            point.reals = [-1.0]
            self.problem.validate(point)
        except ValueError:
            pass
        try:
            point.reals = [2.0]
            self.problem.validate(point)
        except ValueError:
            pass
        point.reals = [1.0]
        try:
            point.ints = [-1]
            self.problem.validate(point)
        except ValueError:
            pass
        try:
            point.ints = [2]
            self.problem.validate(point)
        except ValueError:
            pass

    def test_error6(self):
        self.problem=RealProblem3()
        point = pyomo.opt.blackbox.RealVars()
        point.vars = [1.0]*4
        self.problem.validate(point)
        try:
            point.vars = [-1000.0]*4
            self.problem.validate(point)
        except ValueError:
            pass
        try:
            point.vars = [2000.0]*4
            self.problem.validate(point)
        except ValueError:
            pass
        try:
            point.vars = [0.0]
            self.problem.validate(point)
        except ValueError:
            pass


class TestPoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def test_mi(self):
        point = pyomo.opt.blackbox.MixedIntVars()
        point.reals = [1.0]
        point.ints = [1.0]
        point.bits = [0]
        with capture_output(currdir+'mi_point.out'):
            point.display()
        _out, _txt = currdir+'mi_point.out', currdir+'mi_point.txt'
        self.assertTrue(cmp(_out, _txt),
                        msg="Files %s and %s differ" % (_out, _txt))

    def test_reals(self):
        point = pyomo.opt.blackbox.RealVars()
        point.vars = [1.0]
        with capture_output(currdir+'real_point.out'):
            point.display()
        _out, _txt = currdir+'real_point.out', currdir+'real_point.txt'
        self.assertTrue(cmp(_out, _txt),
                        msg="Files %s and %s differ" % (_out, _txt))



if __name__ == "__main__":
    unittest.main()
