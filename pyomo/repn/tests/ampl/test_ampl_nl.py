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

from filecmp import cmp
import pyomo.common.unittest as unittest

from pyomo.common.getGSL import find_GSL
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, Param, Block,
    ExternalFunction, value,
)

import pyomo.repn.plugins.ampl.ampl_ as ampl_
gsr = ampl_.generate_standard_repn

thisdir = os.path.dirname(os.path.abspath(__file__))

class TestNLWriter(unittest.TestCase):

    def _cleanup(self, fname):
        for x in (fname, fname+'.row', fname+'.col'):
            print(x)
            try:
                os.remove(x)
            except OSError:
                pass

    def _get_fnames(self):
        class_name, test_name = self.id().split('.')[-2:]
        prefix = os.path.join(thisdir, test_name.replace("test_", "", 1))
        return prefix+".nl.baseline", prefix+".nl.out"

    def test_export_nonlinear_variables(self):
        model = ConcreteModel()
        model.x = Var()
        model.y = Var()
        model.z = Var()
        model.w = Var([1,2,3])
        model.c = Constraint(expr=model.x == model.y**2)

        model.y.fix(3)
        test_fname = "export_nonlinear_variables"
        model.write(
            test_fname,
            format='nl',
            io_options={'symbolic_solver_labels':True}
        )
        with open(test_fname + '.col') as f:
            names = list(map(str.strip, f.readlines()))
        assert "z" not in names # z is not in a constraint
        assert "y" not in names # y is fixed
        assert "x" in names
        self._cleanup(test_fname)
        model.write(
            test_fname,
            format='nl',
            io_options={
                'symbolic_solver_labels':True,
                'export_nonlinear_variables':[model.z]
            }
        )
        with open(test_fname + '.col') as f:
            names = list(map(str.strip, f.readlines()))
        assert "z" in names
        assert "y" not in names
        assert "x" in names
        assert "w[1]" not in names
        assert "w[2]" not in names
        assert "w[3]" not in names
        self._cleanup(test_fname)
        model.write(
            test_fname,
            format='nl',
            io_options={
                'symbolic_solver_labels':True,
                'export_nonlinear_variables':[model.z, model.w]
            }
        )
        with open(test_fname + '.col') as f:
            names = list(map(str.strip, f.readlines()))
        assert "z" in names
        assert "y" not in names
        assert "x" in names
        assert "w[1]" in names
        assert "w[2]" in names
        assert "w[3]" in names

        self._cleanup(test_fname)

        model.write(
            test_fname,
            format='nl',
            io_options={
                'symbolic_solver_labels':True,
                'export_nonlinear_variables':[model.z, model.w[2]]
            }
        )
        with open(test_fname + '.col') as f:
            names = list(map(str.strip, f.readlines()))
        assert "z" in names
        assert "y" not in names
        assert "x" in names
        assert "w[1]" not in names
        assert "w[2]" in names
        assert "w[3]" not in names

        self._cleanup(test_fname)

    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaisesRegex(
            KeyError,
            "'a' is not part of the model",
            model.write, test_fname, format='nl')
        self._cleanup(test_fname)

    def test_var_on_deactivated_block(self):
        model = ConcreteModel()
        model.x = Var()
        model.other = Block()
        model.other.a = Var()
        model.other.deactivate()
        model.c = Constraint(expr=model.other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        model.write(test_fname, format='nl')
        self.assertTrue(cmp(
            test_fname,
            baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
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
        self.assertRaisesRegex(
            KeyError,
            "'other.a' exists within Foo 'other'",
            model.write, test_fname, format='nl')
        self._cleanup(test_fname)

    def _external_model(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")

        m = ConcreteModel()
        m.hypot = ExternalFunction(library=DLL, function="gsl_hypot")
        m.p = Param(initialize=1, mutable=True)
        m.x = Var(initialize=3, bounds=(1e-5,None))
        m.y = Var(initialize=3, bounds=(0,None))
        m.z = Var(initialize=1)
        m.o = Objective(
            expr=m.z**2 * m.hypot(m.p*m.x, m.p+m.y)**2)
        self.assertAlmostEqual(value(m.o), 25.0, 7)
        return m

    def test_external_expression_constant(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")

        m = ConcreteModel()
        m.y = Var(initialize=4, bounds=(0,None))
        m.hypot = ExternalFunction(library=DLL, function="gsl_hypot")
        m.o = Objective(expr=m.hypot(3, m.y))
        self.assertAlmostEqual(value(m.o), 5.0, 7)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertTrue(cmp(
            test_fname,
            baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
        self._cleanup(test_fname)

    def test_external_expression_variable(self):
        m = self._external_model()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertTrue(cmp(
            test_fname,
            baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
        self._cleanup(test_fname)

    def test_external_expression_partial_fixed(self):
        m = self._external_model()
        m.x.fix()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertTrue(cmp(
            test_fname,
            baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
        self._cleanup(test_fname)

    def test_external_expression_fixed(self):
        m = self._external_model()
        m.x.fix()
        m.y.fix()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertTrue(cmp(
            test_fname,
            baseline_fname),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
        self._cleanup(test_fname)

    def test_external_expression_rewrite_fixed(self):
        m = self._external_model()

        baseline_fname, test_fname = self._get_fnames()
        variable_baseline = baseline_fname.replace('rewrite_fixed','variable')
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertTrue(cmp(
            test_fname,
            variable_baseline),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))

        m.x.fix()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                io_options={'symbolic_solver_labels':True})
        partial_baseline = baseline_fname.replace(
            'rewrite_fixed','partial_fixed')
        self.assertTrue(cmp(
            test_fname,
            partial_baseline),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))

        m.y.fix()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                io_options={'symbolic_solver_labels':True})
        fixed_baseline = baseline_fname.replace('rewrite_fixed','fixed')
        self.assertTrue(cmp(
            test_fname,
            fixed_baseline),
            msg="Files %s and %s differ" % (test_fname, baseline_fname))
        self._cleanup(test_fname)

    def test_obj_con_cache(self):
        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=model.x**2 >= 1)
        model.obj = Objective(expr=model.x**2)

        with TempfileManager.new_context() as TMP:
            nl_file = TMP.create_tempfile(suffix='.nl')
            model.write(nl_file, format='nl')
            self.assertFalse(hasattr(model, '_repn'))
            with open(nl_file) as FILE:
                nl_ref = FILE.read()

            nl_file = TMP.create_tempfile(suffix='.nl')
            model._gen_obj_repn = True
            model.write(nl_file)
            self.assertEqual(len(model._repn), 1)
            self.assertIn(model.obj, model._repn)
            obj_repn = model._repn[model.obj]
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            nl_file = TMP.create_tempfile(suffix='.nl')
            del model._repn
            model._gen_obj_repn = None
            model._gen_con_repn = True
            model.write(nl_file)
            self.assertEqual(len(model._repn), 1)
            self.assertIn(model.c, model._repn)
            c_repn = model._repn[model.c]
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            nl_file = TMP.create_tempfile(suffix='.nl')
            del model._repn
            model._gen_obj_repn = True
            model._gen_con_repn = True
            model.write(nl_file)
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            nl_file = TMP.create_tempfile(suffix='.nl')
            model._gen_obj_repn = None
            model._gen_con_repn = None
            model.write(nl_file)
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            nl_file = TMP.create_tempfile(suffix='.nl')
            model._gen_obj_repn = True
            model._gen_con_repn = True
            model.write(nl_file)
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIsNot(obj_repn, model._repn[model.obj])
            self.assertIsNot(c_repn, model._repn[model.c])
            obj_repn = model._repn[model.obj]
            c_repn = model._repn[model.c]
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            nl_file = TMP.create_tempfile(suffix='.nl')
            model._gen_obj_repn = False
            model._gen_con_repn = False
            try:
                def dont_call_gsr(*args, **kwargs):
                    self.fail("generate_standard_repn should not be called")
                ampl_.generate_standard_repn = dont_call_gsr
                model.write(nl_file)
            finally:
                ampl_.generate_standard_repn = gsr
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)

            # Check that repns generated by the LP wrter will be
            # processed correctly
            model._repn[model.c] = c_repn = gsr(model.c.body, quadratic=True)
            model._repn[model.obj] = obj_repn = gsr(
                model.obj.expr, quadratic=True)
            nl_file = TMP.create_tempfile(suffix='.nl')
            try:
                def dont_call_gsr(*args, **kwargs):
                    self.fail("generate_standard_repn should not be called")
                ampl_.generate_standard_repn = dont_call_gsr
                model.write(nl_file)
            finally:
                ampl_.generate_standard_repn = gsr
            self.assertEqual(len(model._repn), 2)
            self.assertIn(model.obj, model._repn)
            self.assertIn(model.c, model._repn)
            self.assertIs(obj_repn, model._repn[model.obj])
            self.assertIs(c_repn, model._repn[model.c])
            with open(nl_file) as FILE:
                nl_test = FILE.read()
            self.assertEqual(nl_ref, nl_test)


if __name__ == "__main__":
    unittest.main()
