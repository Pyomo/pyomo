#
# Unit Tests for BuildAction() Objects
#
# PyomoModel                Base test class
# Simple                    Test scalar parameter
# Array1                    Test arrays of parameters
#

import os
import sys
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.core import *


def action1_fn(model):
    model.A = 4.3

def action2_fn(model, i):
    if i in model.A:
        model.A[i] = value(model.A[i])+i

def action3_fn(model, i):
    if i in model.A.sparse_keys():
        model.A[i] = value(model.A[i])+i


class Simple(unittest.TestCase):

    def setUp(self):
        #
        # Create model instance
        #
        model = AbstractModel()
        model.A = Param(initialize=3.3, mutable=True)
        model.action1 = BuildAction(rule=action1_fn)
        self.instance = model.create()

    def tearDown(self):
        self.instance = None
        if os.path.exists("param.dat"):
            os.remove("param.dat")

    def test_value(self):
        """Check the value of the parameter"""
        tmp = value(self.instance.A.value)
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 4.3 )
        self.assertEqual(value(self.instance.A.value), value(self.instance.A))

    def test_getattr(self):
        """Check the use of the __getattr__ method"""
        self.assertEqual( self.instance.A.value, 4.3)



class Array_Param(unittest.TestCase):

    def test_sparse_param_nodefault(self):
        #
        # Create model instance
        #
        model = AbstractModel()
        model.Z = Set(initialize=[1,3])
        model.A = Param(model.Z, initialize={1:1.3}, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create()

        tmp = value(instance.A[1])
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 2.3 )

    def test_sparse_param_nodefault_sparse_iter(self):
        #
        # Create model instance
        #
        model = AbstractModel()
        model.Z = Set(initialize=[1,3])
        model.A = Param(model.Z, initialize={1:1.3}, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action3_fn)
        instance = model.create()

        tmp = value(instance.A[1])
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 2.3 )

    def test_sparse_param_default(self):
        #
        # Create model instance
        #
        model = AbstractModel()
        model.Z = Set(initialize=[1,3])
        model.A = Param(model.Z, initialize={1:1.3}, default=0, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create()

        tmp = value(instance.A[1])
        self.assertEqual( type(tmp), float)
        self.assertEqual( tmp, 2.3 )


    def test_dense_param(self):
        #
        # Create model instance
        #
        model = AbstractModel()
        model.Z = Set(initialize=[1,3])
        model.A = Param(model.Z, initialize=1.3, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create()

        self.assertEqual( instance.A[1], 2.3)
        self.assertEqual( value(instance.A[3]), 4.3)


if __name__ == "__main__":
    unittest.main()
