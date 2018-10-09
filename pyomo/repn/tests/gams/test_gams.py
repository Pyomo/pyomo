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

from six import StringIO

import pyutilib.th as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (Block, ConcreteModel, Connector, Constraint,
                           Objective, TransformationFactory, Var, exp, log)
from pyomo.repn.plugins.gams_writer import (StorageTreeChecker,
                                            expression_to_string,
                                            split_long_line)

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
        return prefix + ".gams.baseline", prefix + ".gams.out"

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
        components["con4"] = Constraint([1, 2], rule=lambda m, i: model.a == i)

        for key in components:
            model.add_component(key, components[key])

        self._check_baseline(model, file_determinism=2)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
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
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "power(abc, 2.0)")

        expr = log(M.abc**2.0)
        self.assertEqual(str(expr), "log(abc**2.0)")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "log(power(abc, 2.0))")

        expr = log(M.abc**2.0) + 5
        self.assertEqual(str(expr), "log(abc**2.0) + 5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "log(power(abc, 2.0)) + 5")

        expr = exp(M.abc**2.0) + 5
        self.assertEqual(str(expr), "exp(abc**2.0) + 5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "exp(power(abc, 2.0)) + 5")

        expr = log(M.abc**2.0)**4
        self.assertEqual(str(expr), "log(abc**2.0)**4")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "power(log(power(abc, 2.0)), 4)")

        expr = log(M.abc**2.0)**4.5
        self.assertEqual(str(expr), "log(abc**2.0)**4.5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), "log(power(abc, 2.0)) ** 4.5")

    def test_power_function_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            m.x ** -3, tc, lbl, smap=smap), "power(x1, (-3))")
        self.assertEqual(expression_to_string(
            m.x ** 0.33, tc, smap=smap), "x1 ** 0.33")
        self.assertEqual(expression_to_string(
            pow(m.x, 2), tc, smap=smap), "power(x1, 2)")

    def test_fixed_var_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.z.fix(-3)
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            m.x + m.y - m.z, tc, lbl, smap=smap), "x1 + x2 + (-1)*(-3)")
        m.z.fix(-400)
        self.assertEqual(expression_to_string(
            m.z + m.y - m.z, tc, smap=smap), "(-400) + x2 + (-1)*(-400)")
        m.z.fix(8.8)
        self.assertEqual(expression_to_string(
            m.x + m.z - m.y, tc, smap=smap), "x1 + (8.8) + (-1)*x2")
        m.z.fix(-8.8)
        self.assertEqual(expression_to_string(
            m.x * m.z - m.y, tc, smap=smap), "x1*(-8.8) + (-1)*x2")

    def test_gams_connector_in_active_constraint(self):
        m = ConcreteModel()
        m.b1 = Block()
        m.b2 = Block()
        m.b1.x = Var()
        m.b2.x = Var()
        m.b1.c = Connector()
        m.b1.c.add(m.b1.x)
        m.b2.c = Connector()
        m.b2.c.add(m.b2.x)
        m.c = Constraint(expr=m.b1.c == m.b2.c)
        m.o = Objective(expr=m.b1.x)
        os = StringIO()
        with self.assertRaises(RuntimeError):
            m.write(os, format="gams")

    def test_gams_expanded_connectors(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.CON1 = Connector()
        m.CON1.add(m.x, 'v')
        m.CON2 = Connector()
        m.CON2.add(m.y, 'v')
        m.c = Constraint(expr=m.CON1 + m.CON2 >= 10)
        TransformationFactory("core.expand_connectors").apply_to(m)
        m.o = Objective(expr=m.x)
        os = StringIO()
        io_options = dict(symbolic_solver_labels=True)
        m.write(os, format="gams", io_options=io_options)
        # no error if we're here, but check for some identifying string
        self.assertIn("x + y", os.getvalue())

    def test_split_long_line(self):
        pat = "var1 + log(var2 / 9) - "
        line = (pat * 10000) + "x"
        self.assertEqual(split_long_line(line),
                         pat * 3478 + "var1 +\nlog(var2 / 9) - " +
                         pat * 3477 + "var1 +\nlog(var2 / 9) - " +
                         pat * 3043 + "x")

    def test_solver_arg(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x == 2)
        m.o = Objective(expr=m.x)
        os = StringIO()
        m.write(os, format="gams", io_options=dict(solver="gurobi"))
        self.assertIn("option lp=gurobi", os.getvalue())

    def test_negative_float_double_operator(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var(bounds=(0, 6))
        m.c = Constraint(expr=(m.x * m.y * -2) == 0)
        m.c2 = Constraint(expr=m.z ** -1.5 == 0)
        m.o = Objective(expr=m.z)
        m.y.fix(-7)
        m.x.fix(4)
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            m.c.body, tc, smap=smap), "(4)*(-7)*(-2)")
        self.assertEqual(expression_to_string(
            m.c2.body, tc, smap=smap), "x1 ** (-1.5)")


class TestGams_writer(unittest.TestCase):

    def _cleanup(self, fname):
        try:
            os.remove(fname)
        except OSError:
            pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix + ".gams.baseline", prefix + ".gams.out"

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2 * model.x <= 0)
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
                kwds.setdefault('ctype', Foo)
                super(Foo, self).__init__(*args, **kwds)

        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaises(
            RuntimeError,
            model.write, test_fname, format='gams')
        self._cleanup(test_fname)


if __name__ == "__main__":
    unittest.main()
