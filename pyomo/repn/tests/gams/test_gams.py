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
# Test the canonical expressions
#

import os

import pyutilib.th as unittest

from pyomo.environ import *
import pyomo.opt

thisdir = os.path.dirname(os.path.abspath(__file__))


class Test(unittest.TestCase):

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix+".gams.baseline", prefix+".gams.out"

    def _check_baseline(self, model, **kwds):
        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        io_options = {"symbolic_solver_labels": True}
        io_options.update(kwds)
        model.write(test_fname,
                    format="gams",
                    io_options=io_options)
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)

    def _gen_expression(self, terms):
        terms = list(terms)
        expr = 0.0
        for term in terms:
            if type(term) is tuple:
                prodterms = list(term)
                prodexpr = 1.0
                for x in prodterms:
                    prodexpr *= x
                expr += prodexpr
            else:
                expr += term
        return expr

    def test_no_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c,
                 (model.a, model.a), (model.b, model.b), (model.c, model.c),
                 (model.a, model.b), (model.a, model.c), (model.b, model.c)]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_no_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_no_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1,2], rule=lambda m, i: model.a == i)

        for key in components:
            model.add_component(key, components[key])

        self._check_baseline(model, file_determinism=2)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)
        self._check_baseline(model)

    def test_expr_xfrm(self):
        from pyomo.repn.plugins.gams_writer import (
            expression_to_string, StorageTreeChecker)
        from pyomo.core.expr.symbol_map import SymbolMap
        M = ConcreteModel()
        M.abc = Var()

        smap = SymbolMap()
        tc = StorageTreeChecker(M)

        expr = M.abc**2.0
        self.assertEqual(str(expr), "abc**2.0")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "power(abc, 2.0)")

        expr = log( M.abc**2.0 )
        self.assertEqual(str(expr), "log(abc**2.0)")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "log(power(abc, 2.0))")

        expr = log( M.abc**2.0 ) + 5
        self.assertEqual(str(expr), "log(abc**2.0) + 5")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "log(power(abc, 2.0)) + 5")

        expr = exp( M.abc**2.0 ) + 5
        self.assertEqual(str(expr), "exp(abc**2.0) + 5")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "exp(power(abc, 2.0)) + 5")

        expr = log( M.abc**2.0 )**4
        self.assertEqual(str(expr), "log(abc**2.0)**4")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "power(log(power(abc, 2.0)), 4)")

        expr = log( M.abc**2.0 )**4.5
        self.assertEqual(str(expr), "log(abc**2.0)**4.5")
        self.assertEqual(expression_to_string(expr, tc, smap=smap), "log(power(abc, 2.0)) ** 4.5")



class TestGams_writer(unittest.TestCase):

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix+".gams.baseline", prefix+".gams.out"

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            RuntimeError,
            model.write, test_fname, format='gams')
        self._cleanup(test_fname)

    def test_var_on_nonblock(self):
        class Foo(Block().__class__):
            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype',Foo)
                super(Foo,self).__init__(*args, **kwds)

        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            RuntimeError,
            model.write, test_fname, format='gams')
        self._cleanup(test_fname)

if __name__ == "__main__":
    unittest.main()
