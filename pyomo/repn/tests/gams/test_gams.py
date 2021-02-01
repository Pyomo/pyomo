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
from pyomo.environ import (Block, ConcreteModel, Constraint,
                           Objective, TransformationFactory, Var, exp, log,
                           ceil, floor, asin, acos, atan, asinh, acosh, atanh,
                           Binary, quicksum)
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
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

    def test_fixed_linear_expr(self):
        # Note that this checks both that a fixed variable is fixed, and
        # that the resulting model type is correctly classified (in this
        # case, fixing a binary makes this an LP)
        m = ConcreteModel()
        m.y = Var(within=Binary)
        m.y.fix(0)
        m.x = Var(bounds=(0,None))
        m.c1 = Constraint(expr=quicksum([m.y, m.y], linear=True) >= 0)
        m.c2 = Constraint(expr=quicksum([m.x, m.y], linear=True) == 1)
        m.obj = Objective(expr=m.x)
        self._check_baseline(m)

    def test_nested_GDP_with_deactivate(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))

        @m.Disjunct([0, 1])
        def disj(disj, _):
            @disj.Disjunct(['A', 'B'])
            def nested(n_disj, _):
                pass  # Blank nested disjunct

            return disj

        m.choice = Disjunction(expr=[m.disj[0], m.disj[1]])

        m.c = Constraint(expr=m.x ** 2 + m.disj[1].nested['A'].indicator_var >= 1)

        m.disj[0].indicator_var.fix(1)
        m.disj[1].deactivate()
        m.disj[0].nested['A'].indicator_var.fix(1)
        m.disj[0].nested['B'].deactivate()
        m.disj[1].nested['A'].indicator_var.set_value(1)
        m.disj[1].nested['B'].deactivate()
        m.o = Objective(expr=m.x)
        TransformationFactory('gdp.fix_disjuncts').apply_to(m)

        outs = StringIO()
        m.write(outs, format='gams', io_options=dict(solver='dicopt'))
        self.assertIn("USING minlp", outs.getvalue())

    def test_quicksum(self):
        m = ConcreteModel()
        m.y = Var(domain=Binary)
        m.c = Constraint(expr=quicksum([m.y, m.y], linear=True) == 1)
        m.y.fix(1)
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(("x1 + x1", False), expression_to_string(m.c.body, tc, smap=smap))
        m.x = Var()
        m.c2 = Constraint(expr=quicksum([m.x, m.y], linear=True) == 1)
        self.assertEqual(("x2 + x1", False), expression_to_string(m.c2.body, tc, smap=smap))

    def test_quicksum_integer_var_fixed(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(domain=Binary)
        m.c = Constraint(expr=quicksum([m.y, m.y], linear=True) == 1)
        m.o = Objective(expr=m.x ** 2)
        m.y.fix(1)
        outs = StringIO()
        m.write(outs, format='gams')
        self.assertIn("USING nlp", outs.getvalue())

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
            expr, tc, smap=smap), ("power(abc, 2)", False))

        expr = log(M.abc**2.0)
        self.assertEqual(str(expr), "log(abc**2.0)")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), ("log(power(abc, 2))", False))

        expr = log(M.abc**2.0) + 5
        self.assertEqual(str(expr), "log(abc**2.0) + 5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), ("log(power(abc, 2)) + 5", False))

        expr = exp(M.abc**2.0) + 5
        self.assertEqual(str(expr), "exp(abc**2.0) + 5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), ("exp(power(abc, 2)) + 5", False))

        expr = log(M.abc**2.0)**4
        self.assertEqual(str(expr), "log(abc**2.0)**4")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), ("power(log(power(abc, 2)), 4)", False))

        expr = log(M.abc**2.0)**4.5
        self.assertEqual(str(expr), "log(abc**2.0)**4.5")
        self.assertEqual(expression_to_string(
            expr, tc, smap=smap), ("log(power(abc, 2)) ** 4.5", False))

    def test_power_function_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            m.x ** -3, tc, lbl, smap=smap), ("power(x1, (-3))", False))
        self.assertEqual(expression_to_string(
            m.x ** 0.33, tc, smap=smap), ("x1 ** 0.33", False))
        self.assertEqual(expression_to_string(
            pow(m.x, 2), tc, smap=smap), ("power(x1, 2)", False))

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
            m.x + m.y - m.z, tc, lbl, smap=smap), ("x1 + x2 - (-3)", False))
        m.z.fix(-400)
        self.assertEqual(expression_to_string(
            m.z + m.y - m.z, tc, smap=smap), ("(-400) + x2 - (-400)", False))
        m.z.fix(8.8)
        self.assertEqual(expression_to_string(
            m.x + m.z - m.y, tc, smap=smap), ("x1 + 8.8 - x2", False))
        m.z.fix(-8.8)
        self.assertEqual(expression_to_string(
            m.x * m.z - m.y, tc, smap=smap), ("x1*(-8.8) - x2", False))

    def test_dnlp_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            ceil(m.x), tc, lbl, smap=smap), ("ceil(x1)", True))
        self.assertEqual(expression_to_string(
            floor(m.x), tc, lbl, smap=smap), ("floor(x1)", True))
        self.assertEqual(expression_to_string(
            abs(m.x), tc, lbl, smap=smap), ("abs(x1)", True))

    def test_arcfcn_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        tc = StorageTreeChecker(m)
        self.assertEqual(expression_to_string(
            asin(m.x), tc, lbl, smap=smap), ("arcsin(x1)", False))
        self.assertEqual(expression_to_string(
            acos(m.x), tc, lbl, smap=smap), ("arccos(x1)", False))
        self.assertEqual(expression_to_string(
            atan(m.x), tc, lbl, smap=smap), ("arctan(x1)", False))
        with self.assertRaisesRegexp(
                RuntimeError,
                "GAMS files cannot represent the unary function asinh"):
            expression_to_string(asinh(m.x), tc, lbl, smap=smap)
        with self.assertRaisesRegexp(
                RuntimeError,
                "GAMS files cannot represent the unary function acosh"):
            expression_to_string(acosh(m.x), tc, lbl, smap=smap)
        with self.assertRaisesRegexp(
                RuntimeError,
                "GAMS files cannot represent the unary function atanh"):
            expression_to_string(atanh(m.x), tc, lbl, smap=smap)

    def test_gams_arc_in_active_constraint(self):
        m = ConcreteModel()
        m.b1 = Block()
        m.b2 = Block()
        m.b1.x = Var()
        m.b2.x = Var()
        m.b1.c = Port()
        m.b1.c.add(m.b1.x)
        m.b2.c = Port()
        m.b2.c.add(m.b2.x)
        m.c = Arc(source=m.b1.c, destination=m.b2.c)
        m.o = Objective(expr=m.b1.x)
        outs = StringIO()
        with self.assertRaises(RuntimeError):
            m.write(outs, format="gams")

    def test_gams_expanded_arcs(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.CON1 = Port()
        m.CON1.add(m.x, 'v')
        m.CON2 = Port()
        m.CON2.add(m.y, 'v')
        m.c = Arc(source=m.CON1, destination=m.CON2)
        TransformationFactory("network.expand_arcs").apply_to(m)
        m.o = Objective(expr=m.x)
        outs = StringIO()
        io_options = dict(symbolic_solver_labels=True)
        m.write(outs, format="gams", io_options=io_options)
        # no error if we're here, but check for some identifying string
        self.assertIn("x - y", outs.getvalue())

    def test_split_long_line(self):
        pat = "var1 + log(var2 / 9) - "
        line = (pat * 10000) + "x"
        self.assertEqual(split_long_line(line),
                         pat * 3478 + "var1 +\n log(var2 / 9) - " +
                         pat * 3477 + "var1 +\n log(var2 / 9) - " +
                         pat * 3043 + "x")

    def test_split_long_line_no_comment(self):
        pat = "1000 * 2000 * "
        line = pat * 5715 + "x"
        self.assertEqual(split_long_line(line),
            pat * 5714 + "1000\n * 2000 * x")

    def test_solver_arg(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x == 2)
        m.o = Objective(expr=m.x)
        outs = StringIO()
        m.write(outs, format="gams", io_options=dict(solver="gurobi"))
        self.assertIn("option lp=gurobi", outs.getvalue())

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
            m.c.body, tc, smap=smap), ("4*(-7)*(-2)", False))
        self.assertEqual(expression_to_string(
            m.c2.body, tc, smap=smap), ("x1 ** (-1.5)", False))


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
