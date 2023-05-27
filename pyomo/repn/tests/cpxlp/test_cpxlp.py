#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the canonical expressions
#

import os
import random

from ..lp_diff import load_and_compare_lp_baseline

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap

thisdir = this_file_dir()


class _CPXLPOrdering_Suite(object):
    @classmethod
    def setUpClass(cls):
        cls.context = TempfileManager.new_context()
        cls.tempdir = cls.context.create_tempdir()

    @classmethod
    def tearDownClass(cls):
        cls.context.release(remove=False)

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = test_name.replace("test_", "", 1)
        return (
            os.path.join(thisdir, prefix + ".lp.baseline"),
            os.path.join(self.tempdir, prefix + ".lp.out"),
        )

    def _check_baseline(self, model, **kwds):
        baseline, testfile = self._get_fnames()
        io_options = {"symbolic_solver_labels": True}
        io_options.update(kwds)
        model.write(testfile, format=self._lp_version, io_options=io_options)
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline, testfile, self._lp_version)
        )

    # Note that this used to generate a random permutation of the
    # expression terms.  However, the default variable ordering in LPv2
    # is the order in which variables are encountered when walking
    # expressions.  Removing the randomization does not significantly
    # change the intent of the test, as the raw term list does not
    # correspond to the final term sequence in the LP tile.
    def _gen_expression(self, terms):
        expr = 0.0
        for term in terms:
            if type(term) is tuple:
                prodexpr = 1.0
                for x in term:
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

        terms = [
            model.a,
            model.b,
            model.c,
            (model.a, model.a),
            (model.b, model.b),
            (model.c, model.c),
            (model.a, model.b),
            (model.a, model.c),
            (model.b, model.c),
        ]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_quadratic(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [
            model.a,
            model.b,
            model.c,
            (model.a, model.a),
            (model.b, model.b),
            (model.c, model.c),
            (model.a, model.b),
            (model.a, model.c),
            (model.b, model.c),
        ]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        # reverse the symbolic ordering
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        self._check_baseline(model)

    def test_column_ordering_linear(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        model.c = Var()

        terms = [model.a, model.b, model.c]
        model.obj = Objective(expr=self._gen_expression(terms))
        model.con = Constraint(expr=self._gen_expression(terms) <= 1)
        # reverse the symbolic ordering
        column_order = ComponentMap()
        column_order[model.a] = 2
        column_order[model.b] = 1
        column_order[model.c] = 0
        self._check_baseline(model, column_order=column_order)

    def test_no_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1, 2], rule=lambda m, i: model.a == i)

        # add components in random order
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])

        self._check_baseline(model, file_determinism=2)

    def test_row_ordering(self):
        model = ConcreteModel()
        model.a = Var()

        components = {}
        components["obj"] = Objective(expr=model.a)
        components["con1"] = Constraint(expr=model.a >= 0)
        components["con2"] = Constraint(expr=model.a <= 1)
        components["con3"] = Constraint(expr=(0, model.a, 1))
        components["con4"] = Constraint([1, 2], rule=lambda m, i: model.a == i)

        # add components in random order
        random_order = list(components.keys())
        random.shuffle(random_order)
        for key in random_order:
            model.add_component(key, components[key])

        # reverse the symbol and index order
        row_order = ComponentMap()
        row_order[model.con1] = 100
        row_order[model.con2] = 2
        row_order[model.con3] = 1
        row_order[model.con4[1]] = 0
        row_order[model.con4[2]] = -1
        self._check_baseline(model, row_order=row_order)


class Test_CPXLPOrdering_v1(_CPXLPOrdering_Suite, unittest.TestCase):
    _lp_version = 'lp_v1'


class Test_CPXLPOrdering_v2(_CPXLPOrdering_Suite, unittest.TestCase):
    _lp_version = 'lp_v2'


class TestCPXLP_writer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TempfileManager.new_context()
        cls.tempdir = cls.context.create_tempdir()

    @classmethod
    def tearDownClass(cls):
        cls.context.release(remove=False)

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = test_name.replace("test_", "", 1)
        return (
            os.path.join(thisdir, prefix + ".lp.baseline"),
            os.path.join(self.tempdir, prefix + ".lp.out"),
        )

    def test_linear_var_on_other_model(self):
        baseline_fname, test_fname = self._get_fnames()

        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)

        # Test var in linear expression
        model.c = Constraint(expr=other.a + 2 * model.x <= 0)
        with LoggingIntercept() as LOG:
            self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
        self.assertEqual(
            LOG.getvalue().replace('\n', ' ').strip(),
            'Model contains an expression (c) that contains a variable '
            '(a) that is not attached to an active block on the '
            'submodel being written',
        )

        # OK with LPv2
        model.write(test_fname, format='lp_v2')
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2')
        )

    def test_quadratic_var_on_other_model(self):
        baseline_fname, test_fname = self._get_fnames()

        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.obj = Objective(expr=model.x)

        # Test var in quadratic expression
        model.c = Constraint(expr=other.a * model.x <= 0)
        with LoggingIntercept() as LOG:
            self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')
        self.assertEqual(
            LOG.getvalue().replace('\n', ' ').strip(),
            'Model contains an expression (c) that contains a variable '
            '(a) that is not attached to an active block on the '
            'submodel being written',
        )

        # OK with LPv2
        model.write(test_fname, format='lp_v2')
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2')
        )

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        model.write(test_fname, format='lp_v1')
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v1')
        )

        model.write(test_fname, format='lp_v2')
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2')
        )

    def test_var_on_nonblock(self):
        class Foo(Block().__class__):
            def __init__(self, *args, **kwds):
                kwds.setdefault('ctype', Foo)
                super(Foo, self).__init__(*args, **kwds)

        model = ConcreteModel()
        model.x = Var()
        model.other = Foo()
        model.other.deactivate()
        model.other.a = Var()
        model.c = Constraint(expr=model.other.a + 2 * model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self.assertRaises(KeyError, model.write, test_fname, format='lp_v1')

        # OK with LPv2
        model.write(test_fname, format='lp_v2')
        self.assertEqual(
            *load_and_compare_lp_baseline(baseline_fname, test_fname, 'lp_v2')
        )

    def test_obj_con_cache(self):
        # Note that the repn caching is only implemented for the v1 writer
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x >= 1)
        model.obj = Objective(expr=model.x * 2)

        with TempfileManager.new_context() as TMP:
            lp_file = TMP.create_tempfile(suffix='.lp')
            model.write(lp_file, format='lp_v1')
            self.assertFalse(hasattr(model, '_repn'))
            with open(lp_file) as FILE:
                lp_ref = FILE.read()

            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 1)
            self.assertIn(model.obj, model._repn)
            obj_repn = model._repn[model.obj]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)

            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = None
            model._gen_con_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)

            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = None
            model._gen_con_repn = None
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)

            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = True
            model._gen_con_repn = True
            model.write(lp_file, format='lp_v1')
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIsNot(obj_repn, model._repn[model.obj])
            self.assertIsNot(c_repn, model._repn[model.c])
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)

            lp_file = TMP.create_tempfile(suffix='.lp')
            model._gen_obj_repn = False
            model._gen_con_repn = False
            import pyomo.repn.plugins.ampl.ampl_ as ampl_

            gsr = ampl_.generate_standard_repn
            try:

                def dont_call_gsr(*args, **kwargs):
                    self.fail("generate_standard_repn should not be called")

                ampl_.generate_standard_repn = dont_call_gsr
                model.write(lp_file, format='lp_v1')
            finally:
                ampl_.generate_standard_repn = gsr
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(lp_file) as FILE:
                lp_test = FILE.read()
            self.assertEqual(lp_ref, lp_test)


if __name__ == "__main__":
    unittest.main()
