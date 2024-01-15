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

import re
import os
from os.path import abspath, dirname, join

currdir = dirname(abspath(__file__))

from filecmp import cmp
import pyomo.common.unittest as unittest

from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
    AbstractModel,
    ConcreteModel,
    Block,
    Set,
    Param,
    Var,
    Objective,
    Constraint,
    Reals,
    display,
)
from pyomo.common.tee import capture_output

from io import StringIO


def rule1(model):
    return (1, model.x + model.y[1], 2)


def rule2(model, i):
    return (1, model.x + model.y[1] + i, 2)


solvers = None


class PyomoModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ

        solvers = check_available_solvers('glpk', 'cplex')

    def test_construct(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.A = Param(initialize=1)
        model.B = Param(model.a)
        model.x = Var(initialize=1, within=Reals, dense=False)
        model.y = Var(model.a, initialize=1, within=Reals, dense=False)
        model.obj = Objective(rule=lambda model: model.x + model.y[1])
        model.obj2 = Objective(model.a, rule=lambda model, i: i + model.x + model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create_instance()
        expr = instance.x + 1

        OUTPUT = open(join(currdir, "display.out"), "w")
        display(instance, ostream=OUTPUT)
        display(instance.obj, ostream=OUTPUT)
        display(instance.x, ostream=OUTPUT)
        display(instance.con, ostream=OUTPUT)
        OUTPUT.write(expr.to_string())
        model = AbstractModel()
        instance = model.create_instance()
        display(instance, ostream=OUTPUT)
        OUTPUT.close()
        try:
            display(None)
            self.fail("test_construct - expected TypeError")
        except TypeError:
            pass
        _out, _txt = join(currdir, "display.out"), join(currdir, "display.txt")
        self.assertTrue(cmp(_out, _txt), msg="Files %s and %s differ" % (_out, _txt))

    def test_construct2(self):
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.A = Param(initialize=1)
        model.B = Param(model.a)
        model.x = Var(initialize=1, within=Reals, dense=True)
        model.y = Var(model.a, initialize=1, within=Reals, dense=True)
        model.obj = Objective(rule=lambda model: model.x + model.y[1])
        model.obj2 = Objective(model.a, rule=lambda model, i: i + model.x + model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create_instance()
        expr = instance.x + 1

        OUTPUT = open(join(currdir, "display2.out"), "w")
        display(instance, ostream=OUTPUT)
        display(instance.obj, ostream=OUTPUT)
        display(instance.x, ostream=OUTPUT)
        display(instance.con, ostream=OUTPUT)
        OUTPUT.write(expr.to_string())
        model = AbstractModel()
        instance = model.create_instance()
        display(instance, ostream=OUTPUT)
        OUTPUT.close()
        try:
            display(None)
            self.fail("test_construct - expected TypeError")
        except TypeError:
            pass
        _out, _txt = join(currdir, "display2.out"), join(currdir, "display2.txt")
        self.assertTrue(cmp(_out, _txt), msg="Files %s and %s differ" % (_out, _txt))


class PyomoBadModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ

        solvers = check_available_solvers('glpk', 'cplex')

    def pyomo(self, cmd, **kwargs):
        args = ['solve'] + re.split('[ ]+', cmd)
        out = kwargs.get('file', None)
        if out is None:
            out = StringIO()
        with capture_output(out):
            os.chdir(currdir)
            output = main(args)
        if not 'file' in kwargs:
            return output.getvalue()
        return output

    def test_uninstantiated_model_linear(self):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        return  # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = join(base, '_linear.out'), join(base, '.txt')
        self.pyomo('uninstantiated_model_linear.py', file=fout)
        self.assertTrue(cmp(fout, fbase), msg="Files %s and %s differ" % (fout, fbase))

    def test_uninstantiated_model_quadratic(self):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'cplex' in solvers:
            self.skipTest("The 'cplex' executable is not available")
        return  # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = join(base, '_quadratic.out'), join(base, '.txt')
        self.pyomo('uninstantiated_model_quadratic.py --solver=cplex', file=fout)
        self.assertTrue(cmp(fout, fbase), msg="Files %s and %s differ" % (fout, fbase))


class TestApplyIndexedRule(unittest.TestCase):
    def test_rules_with_None_in_set(self):
        def noarg_rule(b):
            b.args = ()

        def onearg_rule(b, i):
            b.args = (i,)

        def twoarg_rule(b, i, j):
            b.args = (i, j)

        m = ConcreteModel()
        m.b1 = Block(rule=noarg_rule)
        self.assertEqual(m.b1.args, ())

        m.b2 = Block([None], rule=onearg_rule)
        self.assertEqual(m.b2[None].args, (None,))

        m.b3 = Block([(None, 1)], rule=twoarg_rule)
        self.assertEqual(m.b3[None, 1].args, ((None, 1)))


class TestComponent(unittest.TestCase):
    def test_getname(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.v = Var()
        self.assertEqual(m.b.v.getname(fully_qualified=True, relative_to=m.b), 'v')

    def test_getname_error(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.v = Var()
        m.c = Block()
        self.assertRaises(
            RuntimeError, m.b.v.getname, fully_qualified=True, relative_to=m.c
        )


if __name__ == "__main__":
    unittest.main()
