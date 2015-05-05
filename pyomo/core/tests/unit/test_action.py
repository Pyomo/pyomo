#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for BuildAction() Objects
#
# PyomoModel                Base test class
# Simple                    Test scalar parameter
# Array1                    Test arrays of parameters
#

from six import StringIO
import os

import pyutilib.th as unittest

from pyomo.environ import *


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
        self.instance = model.create_instance()

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
        instance = model.create_instance()

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
        instance = model.create_instance()

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
        instance = model.create_instance()

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
        instance = model.create_instance()
        #
        self.assertEqual( instance.A[1], 2.3)
        self.assertEqual( value(instance.A[3]), 4.3)
        #
        buf = StringIO()
        instance.pprint(ostream=buf)
        self.assertEqual(buf.getvalue(),"""1 Set Declarations
    Z : Dim=0, Dimen=1, Size=2, Domain=None, Ordered=False, Bounds=(1, 3)
        [1, 3]

1 Param Declarations
    A : Size=2, Index=Z, Domain=Any, Default=None, Mutable=True
        Key : Value
          1 :   2.3
          3 :   4.3

1 BuildAction Declarations
    action2 : Size=0, Index=Z, Active=True

3 Declarations: Z A action2
""")


class TestMisc(unittest.TestCase):

    def test_error1(self):
        model = AbstractModel()
        try:
            model.a = BuildAction()
            self.fail("Expected ValueError")
        except ValueError:
            pass
        

if __name__ == "__main__":
    unittest.main()
