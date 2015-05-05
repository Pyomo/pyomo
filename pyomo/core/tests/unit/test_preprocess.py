#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for model preprocessing
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *

class TestPreprocess(unittest.TestCase):

    def Xtest_label1(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300})
        model.x = Var(model.A)
        model.y = Var(model.A)
        instance=model.create_instance()
        instance.preprocess()
        self.assertEqual(instance.num_used_variables(),0)

    def Xtest_label2(self):
        model = AbstractModel()
        model.A = Set(initialize=[1,2,3])
        model.B = Param(model.A,initialize={1:100,2:200,3:300})
        model.x = Var(model.A)
        model.y = Var(model.A)
        model.obj = Objective(rule=lambda inst: inst.x[1])
        instance=model.create_instance()
        instance.preprocess()
        self.assertEqual(instance.num_used_variables(),1)
        self.assertEqual(instance.x[1].label,"x(1)")
        self.assertEqual(instance.x[2].label,"x(2)")
        self.assertEqual(instance.y[1].label,"y(1)")

if __name__ == "__main__":
    unittest.main()
