#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for BuildCheck() Objects
#
# PyomoModel                Base test class
# Simple                    Test scalar parameter
# Array1                    Test arrays of parameters
#

import os
from six import StringIO

import pyutilib.th as unittest

from pyomo.environ import *

class PyomoModel(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()
        self.instance = None

    def tearDown(self):
        self.model = None
        self.instance = None

    def construct(self,filename):
        self.instance = self.model.create_instance(filename)


def action1a_fn(model):
    return value(model.A) == 3.3

def action1b_fn(model):
    return value(model.A) != 3.3

def action2a_fn(model, i):
    ans=True
    if i in model.A:
        return (value(model.A[i]) == 1.3)
    return True

def action2b_fn(model, i):
    if i in model.A:
        ans = (value(model.A[i]) == 1.3)
        #print "HERE",i,ans,not ans
        return not ans
    return True


class Simple(PyomoModel):

    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        #
        # Create model instance
        #
        self.model.A = Param(initialize=3.3)

    def tearDown(self):
        if os.path.exists("param.dat"):
            os.remove("param.dat")
        PyomoModel.tearDown(self)

    def test_true(self):
        """Apply a build check that returns true"""
        self.model.action1 = BuildCheck(rule=action1a_fn)
        self.instance = self.model.create_instance()
        tmp = value(self.instance.A)
        self.assertEqual( tmp, 3.3 )

    def test_false(self):
        """Apply a build check that returns false"""
        self.model.action1 = BuildCheck(rule=action1b_fn)
        try:
            self.instance = self.model.create_instance()
            self.fail("expected failure")
        except ValueError:
            pass



class Array1(PyomoModel):

    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        #
        # Create model instance
        #
        self.model.Z = Set(initialize=[1,3])
        self.model.A = Param(self.model.Z, initialize=1.3)

    def tearDown(self):
        PyomoModel.tearDown(self)

    def test_true(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2a_fn)
        self.instance = self.model.create_instance()

    def test_false(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2b_fn)
        try:
            self.instance = self.model.create_instance()
            self.fail("expected failure")
        except ValueError:
            pass


class Array2(PyomoModel):

    def setUp(self):
        #
        # Create Model
        #
        PyomoModel.setUp(self)
        #
        # Create model instance
        #
        self.model.Z = Set(initialize=[1,3])
        self.model.A = Param(self.model.Z, initialize=1.3)

    def tearDown(self):
        PyomoModel.tearDown(self)

    def test_true(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2a_fn)
        self.instance = self.model.create_instance()

    def test_false(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2b_fn)
        try:
            self.instance = self.model.create_instance()
            self.fail("expected failure")
        except ValueError:
            pass


class TestMisc(unittest.TestCase):

    def test_error1(self):
        model = AbstractModel()
        try:
            model.a = BuildCheck()
            self.fail("Expected ValueError")
        except ValueError:
            pass

    def test_io(self):
        model = AbstractModel()
        model.c1 = BuildCheck(rule=lambda M: True)
        model.A = Set(initialize=[1,2,3])
        model.c2 = BuildCheck(model.A, rule=lambda M,i: True)
        instance = model.create_instance()
        #
        buf = StringIO()
        instance.pprint(ostream=buf)
        self.assertEqual(buf.getvalue(),"""1 Set Declarations
    A : Dim=0, Dimen=1, Size=3, Domain=None, Ordered=False, Bounds=(1, 3)
        [1, 2, 3]

2 BuildCheck Declarations
    c1 : 
    c2 : 

3 Declarations: c1 A c2
""")



if __name__ == "__main__":
    unittest.main()
