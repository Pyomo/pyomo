#
# Unit Tests for BuildCheck() Objects
#
# PyomoModel                Base test class
# Simple                    Test scalar parameter
# Array1                    Test arrays of parameters
#

import os
import sys
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.environ import *

class PyomoModel(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def construct(self,filename):
        self.instance = self.model.create(filename)


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

    def test_true(self):
        """Apply a build check that returns true"""
        self.model.action1 = BuildCheck(rule=action1a_fn)
        self.instance = self.model.create()
        tmp = value(self.instance.A)
        self.assertEqual( tmp, 3.3 )

    def test_false(self):
        """Apply a build check that returns false"""
        self.model.action1 = BuildCheck(rule=action1b_fn)
        try:
            self.instance = self.model.create()
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

    def test_true(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2a_fn)
        self.instance = self.model.create()

    def test_false(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2b_fn)
        try:
            self.instance = self.model.create()
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

    def test_true(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2a_fn)
        self.instance = self.model.create()

    def test_false(self):
        """Check the value of the parameter"""
        self.model.action2 = BuildCheck(self.model.Z, rule=action2b_fn)
        try:
            self.instance = self.model.create()
            self.fail("expected failure")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main()
