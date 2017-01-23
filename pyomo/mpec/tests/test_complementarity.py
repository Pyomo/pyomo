#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Test different transformations for complementarity conditions.
#
# These perform difference tests with files that capture the structure of
# the transformed model.
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__)) + os.sep

import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect

from pyomo.opt import ProblemFormat
from pyomo.core import *
from pyomo.mpec import Complementarity, complements, ComplementarityList

class CCTests(object):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def _setup(self):
        M = ConcreteModel()
        M.y = Var()
        M.x1 = Var()
        M.x2 = Var()
        M.x3 = Var()
        return M

    def _print(self, model):
        model.cc.pprint()

    def _test(self, tname, M):
        ofile = currdir + tname + '_%s.out' % str(self.xfrm)
        bfile = currdir + tname + '_%s.txt' % str(self.xfrm)
        if self.xfrm is not None:
            xfrm = TransformationFactory(self.xfrm)
            xfrm.apply_to(M)
        setup_redirect(ofile)
        self._print(M)
        reset_redirect()
        if not os.path.exists(bfile):
            os.rename(ofile, bfile)
        self.assertFileEqualsBaseline(ofile, bfile)

    def test_t1a(self):
        # y + x1 >= 0  _|_  x1 + 2*x2 + 3*x3 >= 1
        M = self._setup()
        M.c = Constraint(expr=M.y + M.x3 >= M.x2)
        M.cc = Complementarity(expr=complements(M.y + M.x1 >= 0, M.x1 + 2*M.x2 + 3*M.x3 >= 1))
        self._test("t1a", M)

    def test_t1b(self):
        # Reversing the expressions in test t1a:
        #    x1 + 2*x2 + 3*x3 >= 1  _|_  y + x1 >= 0
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2*M.x2 + 3*M.x3 >= 1, M.y + M.x1 >= 0))
        self._test("t1b", M)

    def test_t1c(self):
        # y >= - x1  _|_  x1 + 2*x2 >= 1 - 3*x3
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y >= - M.x1, M.x1 + 2*M.x2 >= 1 - 3*M.x3))
        self._test("t1c", M)


    def test_t2a(self):
        # y + x2 >= 0  _|_  x2 - x3 <= -1
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x2 >= 0, M.x2 - M.x3 <= -1))
        self._test("t2a", M)

    def test_t2b(self):
        # Reversing the expressions in test t2a:
        #    x2 - x3 <= -1  _|_  y + x2 >= 0
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x2 - M.x3 <= -1, M.y + M.x2 >= 0))
        self._test("t2b", M)


    def test_t3a(self):
        # y + x3 >= 0  _|_  x1 + x2 >= -1
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3 >= 0, M.x1 + M.x2 >= -1))
        self._test("t3a", M)

    def test_t3b(self):
        # Reversing the expressions in test t3a:
        #    x1 + x2 >= -1  _|_  y + x3 >= 0
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + M.x2 >= -1, M.y + M.x3 >= 0))
        self._test("t3b", M)


    def test_t4a(self):
        # x1 + 2*x2 + 3*x3 = 1  _|_  y + x3
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2*M.x2 + 3*M.x3 == 1, M.y + M.x3))
        self._test("t4a", M)

    def test_t4b(self):
        # Reversing the expressions in test t7b:
        #    y + x3  _|_  x1 + 2*x2 + 3*x3 = 1
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3, M.x1 + 2*M.x2 + 3*M.x3 == 1))
        self._test("t4b", M)

    def test_t4c(self):
        # 1 = x1 + 2*x2 + 3*x3  _|_  y + x3
        M = self._setup()
        M.cc = Complementarity(expr=complements(1 == M.x1 + 2*M.x2 + 3*M.x3, M.y + M.x3))
        self._test("t4c", M)

    def test_t4d(self):
        # x1 + 2*x2 == 1 - 3*x3  _|_  y + x3
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1 + 2*M.x2 == 1 - 3*M.x3, M.y + M.x3))
        self._test("t4d", M)


    def test_t9(self):
        # Testing that we can skip deactivated complementarity conditions
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y + M.x3, M.x1 + 2*M.x2 == 1))
        M.cc.deactivate()
        self._test("t9", M)

    def test_t10(self):
        # Testing that we can skip an array of deactivated complementarity conditions
        M = self._setup()
        def f(model, i):
            return complements(M.y + M.x3, M.x1 + 2*M.x2 == i)
        M.cc = Complementarity([0,1,2], rule=f)
        M.cc[1].deactivate()
        self._test("t10", M)

    def test_t11(self):
        # 2 <= y + x1 <= 3  _|_  x1
        M = self._setup()
        M.cc = Complementarity(expr=complements(2 <= M.y + M.x1 <= 3, M.x1))
        self._test("t11", M)

    def test_t12(self):
        # x1  _|_  2 <= y + x1 <= 3"""
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.x1, 2 <= M.y + M.x1 <= 3))
        self._test("t12", M)

    def test_t13(self):
        # Testing that we can skip an array of deactivated complementarity conditions
        M = self._setup()
        def f(model, i):
            if i == 0:
                return complements(M.y + M.x3, M.x1 + 2*M.x2 == 0)
            if i == 1:
                return Complementarity.Skip
            if i == 2:
                return complements(M.y + M.x3, M.x1 + 2*M.x2 == 2)
        M.cc = Complementarity([0,1,2], rule=f)
        self._test("t13", M)

    def test_cov2(self):
        # Testing warning for no rule"""
        M = self._setup()
        M.cc = Complementarity([0,1,2])
        self._test("cov2", M)

    def test_cov4(self):
        # Testing construction with no indexing and a rule
        M = self._setup()
        def f(model):
            return complements(M.y + M.x3, M.x1 + 2*M.x2 == 1)
        M.cc = Complementarity(rule=f)
        self._test("cov4", M)

    def test_cov5(self):
        # Testing construction with rules that generate an exception
        M = self._setup()
        def f(model):
            raise IOError("cov5 error")
        try:
            M.cc1 = Complementarity(rule=f)
            self.fail("Expected an IOError")
        except IOError:
            pass
        def f(model, i):
            raise IOError("cov5 error")
        try:
            M.cc2 = Complementarity([0,1], rule=f)
            self.fail("Expected an IOError")
        except IOError:
            pass

    def test_cov6(self):
        # Testing construction with indexing and an expression
        M = self._setup()
        try:
            M.cc = Complementarity([0,1], expr=())
            self.fail("Expected an IndexError")
        except IndexError:
            pass

    def test_cov7(self):
        # Testing error checking with return value
        M = self._setup()
        def f(model):
            return ()
        try:
            M.cc = Complementarity(rule=f)
            self.fail("Expected ValueError")
        except ValueError:
            pass
        def f(model):
            return
        try:
            M.cc = Complementarity(rule=f)
            self.fail("Expected ValueError")
        except ValueError:
            pass
        def f(model):
            return {}
        try:
            M.cc = Complementarity(rule=f)
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_cov8(self):
        # Testing construction with a list
        M = self._setup()
        def f(model):
            return [M.y + M.x3, M.x1 + 2*M.x2 == 1]
        M.cc = Complementarity(rule=f)
        self._test("cov8", M)

    def test_cov9(self):
        # Testing construction with a tuple
        M = self._setup()
        def f(model):
            return (M.y + M.x3, M.x1 + 2*M.x2 == 1)
        M.cc = Complementarity(rule=f)
        self._test("cov8", M)

    def test_cov10(self):
        # Testing construction with a badly formed expression
        M = self._setup()
        M.cc = Complementarity(expr=complements(M.y <= M.x1 <= 1, M.x2))
        try:
            M.cc.to_standard_form()
            self.fail("Expected a RuntimeError")
        except RuntimeError:
            pass

    def test_cov11(self):
        # Testing construction with a badly formed expression
        M = self._setup()
        M.cc = Complementarity(expr=complements(1 <= M.x1 <= M.y, M.x2))
        try:
            M.cc.to_standard_form()
            self.fail("Expected a RuntimeError")
        except RuntimeError:
            pass

    def test_list1(self):
        M = self._setup()
        M.cc = ComplementarityList()
        M.cc.add( complements(M.y + M.x3, M.x1 + 2*M.x2 == 0) )
        M.cc.add( complements(M.y + M.x3, M.x1 + 2*M.x2 == 2) )
        self._test("list1", M)

    def test_list2(self):
        M = self._setup()
        M.cc = ComplementarityList()
        M.cc.add( complements(M.y + M.x3, M.x1 + 2*M.x2 == 0) )
        M.cc.add( complements(M.y + M.x3, M.x1 + 2*M.x2 == 1) )
        M.cc.add( complements(M.y + M.x3, M.x1 + 2*M.x2 == 2) )
        M.cc[2].deactivate()
        self._test("list2", M)

    def test_list3(self):
        M = self._setup()
        def f(M, i):
            if i == 1:
                return complements(M.y + M.x3, M.x1 + 2*M.x2 == 0)
            elif i == 2:
                return complements(M.y + M.x3, M.x1 + 2*M.x2 == 2)
            return ComplementarityList.End
        M.cc = ComplementarityList(rule=f)
        self._test("list1", M)

    def test_list4(self):
        M = self._setup()
        def f(M):
            yield complements(M.y + M.x3, M.x1 + 2*M.x2 == 0)
            yield complements(M.y + M.x3, M.x1 + 2*M.x2 == 2)
            yield ComplementarityList.End
        M.cc = ComplementarityList(rule=f)
        self._test("list1", M)

    def test_list5(self):
        M = self._setup()
        M.cc = ComplementarityList(rule=(complements(M.y + M.x3, M.x1 + 2*M.x2 == i) for i in range(3)))
        self._test("list5", M)

    def test_list6(self):
        M = self._setup()
        try:
            M.cc = ComplementarityList()
            self.fail("Expected a RuntimeError")
        except:
            pass

    def test_list7(self):
        M = self._setup()
        def f(M):
            return None
        try:
            M.cc = ComplementarityList(rule=f)
            self.fail("Expected a ValueError")
        except:
            pass
        M = self._setup()
        def f(M):
            yield None
        try:
            M.cc = ComplementarityList(rule=f)
            self.fail("Expected a ValueError")
        except:
            pass


class CCTests_none(CCTests, unittest.TestCase):

    xfrm = None


class CCTests_nl(CCTests, unittest.TestCase):

    xfrm = 'mpec.nl'

    def _print(self, model):
        model.pprint()


class CCTests_standard_form(CCTests, unittest.TestCase):

    xfrm = 'mpec.standard_form'


class CCTests_simple_nonlinear(CCTests, unittest.TestCase):

    xfrm = 'mpec.simple_nonlinear'


class CCTests_simple_disjunction(CCTests, unittest.TestCase):

    xfrm = 'mpec.simple_disjunction'


class CCTests_nl_nlxfrm(CCTests, unittest.TestCase):

    def _test(self, tname, M):
        ofile = currdir + tname + '_nlxfrm.out'
        bfile = currdir + tname + '_nlxfrm.nl'
        xfrm = TransformationFactory('mpec.nl')
        xfrm.apply_to(M)
        M.write(ofile, format=ProblemFormat.nl)
        if not os.path.exists(bfile):
            os.rename(ofile, bfile)
        self.assertFileEqualsBaseline(ofile, bfile)


if __name__ == "__main__":
    unittest.main()
