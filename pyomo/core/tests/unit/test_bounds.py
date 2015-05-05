#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for nontrivial Bounds (_SumExpression, _ProductExpression)
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import *

class Test(unittest.TestCase):

    #Test constraint bounds
    def test_constr_lower(self):
        model = AbstractModel()
        model.A = Param(default=2.0, mutable=True)
        model.B = Param(default=1.5, mutable=True)
        model.C = Param(default=2.5, mutable=True)
        model.X = Var()

        def constr_rule(model):
            return (model.A*(model.B+model.C),model.X)
        model.constr = Constraint(rule=constr_rule)

        instance = model.create_instance()
        self.assertEqual(instance.constr.lower(),8.0)

    def test_constr_upper(self):
        model = AbstractModel()
        model.A = Param(default=2.0, mutable=True)
        model.B = Param(default=1.5, mutable=True)
        model.C = Param(default=2.5, mutable=True)
        model.X = Var()

        def constr_rule(model):
            return (model.X,model.A*(model.B+model.C))
        model.constr = Constraint(rule=constr_rule)

        instance = model.create_instance()

        self.assertEqual(instance.constr.upper(),8.0)

    def test_constr_both(self):
        model = AbstractModel()
        model.A = Param(default=2.0, mutable=True)
        model.B = Param(default=1.5, mutable=True)
        model.C = Param(default=2.5, mutable=True)
        model.X = Var()

        def constr_rule(model):
            return (model.A*(model.B-model.C),model.X,model.A*(model.B+model.C))
        model.constr = Constraint(rule=constr_rule)

        instance = model.create_instance()

        self.assertEqual(instance.constr.lower(),-2.0)
        self.assertEqual(instance.constr.upper(),8.0)


    #Test variable bounds
    #JPW: Disabled until we are convinced that we want to support complex parametric expressions for variable bounds.
    def test_var_bounds(self):
        model = AbstractModel()
        model.A = Param(default=2.0, mutable=True)
        model.B = Param(default=1.5, mutable=True)
        model.C = Param(default=2.5)

        def X_bounds_rule(model):
            return (model.A*(model.B-model.C),model.A*(model.B+model.C))
        model.X = Var(bounds=X_bounds_rule)

        instance = model.create_instance()

        self.assertEqual(instance.X.lb,-2.0)
        self.assertEqual(instance.X.ub,8.0)
   

if __name__ == "__main__":
    unittest.main()

