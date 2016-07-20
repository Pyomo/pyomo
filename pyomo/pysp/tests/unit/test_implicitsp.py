#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
import pyutilib.th as unittest

import pyomo.core as pc
from pyomo.pysp.implicitsp import ImplicitSP

class TestImplicitSP(unittest.TestCase):

    def test_collect_mutable_parameters(self):
        model = pc.ConcreteModel()
        model.p = pc.Param(mutable=True)
        model.q = pc.Param([1], mutable=True, initialize=1.0)
        model.r = pc.Param(initialize=1.1, mutable=False)
        model.x = pc.Var()
        for obj in [model.p, model.q[1]]:

            result = ImplicitSP._collect_mutable_parameters(
                obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                2 * (obj + 1))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                2 * obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                2 * obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                2 * obj + 1 + model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                obj * model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                model.x / obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                model.x / (2 * obj))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                obj * pc.log(2 * model.x))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                obj * pc.sin(model.r) ** model.x)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_mutable_parameters(
                model.x**(obj * pc.sin(model.r)))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

        result = ImplicitSP._collect_mutable_parameters(
            1.0)
        self.assertEqual(len(result), 0)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.p + model.q[1] + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.p + 1 + model.r + model.q[1])
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)

        result = ImplicitSP._collect_mutable_parameters(
            model.q[1] * 2 * (model.p + model.r) + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            2 * model.x * model.p * model.q[1] * model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            2 * obj * model.q[1] * model.r + 1)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 1)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            2 * model.q[1] + 1 + model.x - model.p)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.r * model.x)
        self.assertEqual(len(result), 0)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.x / obj)
        self.assertTrue(id(obj) in result)
        self.assertEqual(len(result), 1)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.x / (2 * model.q[1] / model.p))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            (model.p / model.q[1]) * pc.log(2 * model.x))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            model.q[1] * pc.sin(model.p) ** (model.x + model.r))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_mutable_parameters(
            (model.p + model.x) ** (model.q[1] * pc.sin(model.r)))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

    def test_collect_variables(self):
        model = pc.ConcreteModel()
        model.p = pc.Var()
        model.p.fixed = True
        model.q = pc.Var([1])
        model.r = pc.Param(mutable=True)
        model.x = pc.Var()
        for obj in [model.p, model.q[1]]:

            result = ImplicitSP._collect_variables(
                obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_variables(
                obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_variables(
                2 * (obj + 1))
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_variables(
                2 * obj)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_variables(
                2 * obj + 1)
            self.assertTrue(id(obj) in result)
            self.assertEqual(len(result), 1)
            del result

            result = ImplicitSP._collect_variables(
                2 * obj + 1 + model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                obj * model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                model.x / obj)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                model.x / (2 * obj))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                obj * pc.log(2 * model.x))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                obj * pc.sin(model.r) ** model.x)
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

            result = ImplicitSP._collect_variables(
                model.x**(obj * pc.sin(model.r)))
            self.assertTrue(id(obj) in result)
            self.assertTrue(id(model.x) in result)
            self.assertEqual(len(result), 2)
            del result

        result = ImplicitSP._collect_variables(
            1.0)
        self.assertEqual(len(result), 0)
        del result

        result = ImplicitSP._collect_variables(
            model.p + model.q[1] + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_variables(
            model.p + 1 + model.r + model.q[1])
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)

        result = ImplicitSP._collect_variables(
            model.q[1] * 2 * (model.p + model.r) + model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_variables(
            2 * model.x * model.p * model.q[1] * model.r)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = ImplicitSP._collect_variables(
            2 * obj * model.q[1] * model.r + 1)
        self.assertTrue(id(model.q[1]) in result)
        self.assertEqual(len(result), 1)
        del result

        result = ImplicitSP._collect_variables(
            2 * model.q[1] + 1 + model.x - model.p)
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = ImplicitSP._collect_variables(
            model.r * model.x)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 1)
        del result

        result = ImplicitSP._collect_variables(
            model.x / obj)
        self.assertTrue(id(obj) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 2)
        del result

        result = ImplicitSP._collect_variables(
            model.x / (2 * model.q[1] / model.p))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = ImplicitSP._collect_variables(
            (model.p / model.q[1]) * pc.log(2 * model.x))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = ImplicitSP._collect_variables(
            model.q[1] * pc.sin(model.p) ** (model.x + model.r))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

        result = ImplicitSP._collect_variables(
            (model.p + model.x) ** (model.q[1] * pc.sin(model.r)))
        self.assertTrue(id(model.p) in result)
        self.assertTrue(id(model.q[1]) in result)
        self.assertTrue(id(model.x) in result)
        self.assertEqual(len(result), 3)
        del result

TestImplicitSP = unittest.category('smoke','nightly','expensive')(TestImplicitSP)

if __name__ == "__main__":
    unittest.main()
