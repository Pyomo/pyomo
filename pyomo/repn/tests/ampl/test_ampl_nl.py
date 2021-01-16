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

from pyomo.common.getGSL import find_GSL
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Param, Block, ExternalFunction, value

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


    def test_var_on_other_model(self):
        other = ConcreteModel()
        other.a = Var()

        model = ConcreteModel()
        model.x = Var()
        model.c = Constraint(expr=other.a + 2*model.x <= 0)
        model.obj = Objective(expr=model.x)

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        self.assertRaisesRegexp(
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
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)

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
        self.assertRaisesRegexp(
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

    def test_external_expression_variable(self):
        m = self._external_model()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)
        self._cleanup(test_fname)

    def test_external_expression_partial_fixed(self):
        m = self._external_model()
        m.x.fix()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)
        self._cleanup(test_fname)

    def test_external_expression_fixed(self):
        m = self._external_model()
        m.x.fix()
        m.y.fix()

        baseline_fname, test_fname = self._get_fnames()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertFileEqualsBaseline(
            test_fname,
            baseline_fname,
            delete=True)
        self._cleanup(test_fname)

    def test_external_expression_rewrite_fixed(self):
        m = self._external_model()

        baseline_fname, test_fname = self._get_fnames()
        variable_baseline = baseline_fname.replace('rewrite_fixed','variable')
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                    io_options={'symbolic_solver_labels':True})
        self.assertFileEqualsBaseline(
            test_fname,
            variable_baseline,
            delete=True)

        self.assertIsNot(m._repn, None)

        m.x.fix()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                io_options={'symbolic_solver_labels':True})
        partial_baseline = baseline_fname.replace(
            'rewrite_fixed','partial_fixed')
        self.assertFileEqualsBaseline(
            test_fname,
            partial_baseline,
            delete=True)

        m.y.fix()
        self._cleanup(test_fname)
        m.write(test_fname, format='nl',
                io_options={'symbolic_solver_labels':True})
        fixed_baseline = baseline_fname.replace('rewrite_fixed','fixed')
        self.assertFileEqualsBaseline(
            test_fname,
            fixed_baseline,
            delete=True)
        self._cleanup(test_fname)


if __name__ == "__main__":
    unittest.main()
