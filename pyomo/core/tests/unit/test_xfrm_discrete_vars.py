#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for Discrete Variable Transformations

import pyomo.common.unittest as unittest

from pyomo.core.base import VarCollector
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Objective,
    Suffix,
    Binary,
    TransformationFactory,
    SolverFactory,
    Reals,
    Block,
    Integers,
    value,
)
from pyomo.opt import check_available_solvers

solvers = check_available_solvers('cplex', 'gurobi', 'glpk')


def _generateModel():
    model = ConcreteModel()
    model.x = Var(within=Binary)
    model.y = Var()
    model.c1 = Constraint(expr=model.y >= model.x)
    model.c2 = Constraint(expr=model.y >= 1.5 - model.x)
    model.obj = Objective(expr=model.y)
    model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    return model


def _make_hierarchical_model():
    m = ConcreteModel()
    m.y = Var(domain=Binary)
    m.b = Block()
    m.b.x = Var([1, 2], bounds=(2, 45), domain=Integers)
    m.b.y = Var(domain=Binary)
    m.b.c = Constraint(expr=m.b.x[1] * m.y <= 23)

    return m


class Test(unittest.TestCase):
    @unittest.skipIf(len(solvers) == 0, "LP/MIP solver not available")
    def test_solve_relax_transform(self):
        s = SolverFactory(solvers[0])
        m = _generateModel()
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 0)

        TransformationFactory('core.relax_integer_vars').apply_to(m)
        self.assertIs(m.x.domain, Reals)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 2)
        self.assertAlmostEqual(m.dual[m.c1], -0.5, 4)
        self.assertAlmostEqual(m.dual[m.c2], -0.5, 4)

    def test_reverse_relax_integer_vars(self):
        m = _generateModel()
        lp_relax = TransformationFactory('core.relax_integer_vars')
        reverse = lp_relax.apply_to(m)
        self.assertIs(m.x.domain, Reals)
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        self.assertIsNone(m.y.lb)
        self.assertIsNone(m.y.ub)

        lp_relax.apply_to(m, reverse=reverse)
        self.assertIs(m.x.domain, Binary)
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        self.assertIsNone(m.y.lb)
        self.assertIsNone(m.y.ub)

    def test_relax_integer_vars_block_targets(self):
        m = _make_hierarchical_model()
        TransformationFactory('core.relax_integer_vars').apply_to(m, targets=m.b)
        for i in [1, 2]:
            self.assertIs(m.b.x[i].domain, Reals)
            self.assertEqual(m.b.x[i].lb, 2)
            self.assertEqual(m.b.x[i].ub, 45)
        self.assertIs(m.b.y.domain, Reals)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)

        self.assertIs(m.y.domain, Binary)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

    def test_relax_integer_vars_var_data_targets(self):
        m = _make_hierarchical_model()
        TransformationFactory('core.relax_integer_vars').apply_to(
            m, targets=[m.b.x[1], m.y]
        )
        # transformed
        self.assertIs(m.b.x[1].domain, Reals)
        self.assertEqual(m.b.x[1].lb, 2)
        self.assertEqual(m.b.x[1].ub, 45)
        # not transformed
        self.assertIs(m.b.x[2].domain, Integers)
        self.assertEqual(m.b.x[2].lb, 2)
        self.assertEqual(m.b.x[2].ub, 45)
        # not transformed
        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        # transformed
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

    def test_relax_integer_vars_indexed_var_targets(self):
        m = _make_hierarchical_model()
        TransformationFactory('core.relax_integer_vars').apply_to(m, targets=m.b.x)
        # transformed
        for i in [1, 2]:
            self.assertIs(m.b.x[i].domain, Reals)
            self.assertEqual(m.b.x[i].lb, 2)
            self.assertEqual(m.b.x[i].ub, 45)
        # not transformed
        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        # not transformed
        self.assertIs(m.y.domain, Binary)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

    def test_relax_integer_vars_vars_from_expressions(self):
        m = _make_hierarchical_model()
        TransformationFactory('core.relax_integer_vars').apply_to(
            m.b, var_collector=VarCollector.FromExpressions
        )
        # transformed
        self.assertIs(m.b.x[1].domain, Reals)
        self.assertEqual(m.b.x[1].lb, 2)
        self.assertEqual(m.b.x[1].ub, 45)
        # not transformed
        self.assertIs(m.b.x[2].domain, Integers)
        self.assertEqual(m.b.x[2].lb, 2)
        self.assertEqual(m.b.x[2].ub, 45)
        # not transformed
        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        # transformed
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

    def test_relax_integer_vars_ignore_deactivated_blocks(self):
        m = _make_hierarchical_model()
        m.b.deactivate()
        TransformationFactory('core.relax_integer_vars').apply_to(
            m, transform_deactivated_blocks=False
        )
        # not transformed
        for i in [1, 2]:
            self.assertIs(m.b.x[i].domain, Integers)
            self.assertEqual(m.b.x[i].lb, 2)
            self.assertEqual(m.b.x[i].ub, 45)
        # not transformed
        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        # transformed
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

        m = _make_hierarchical_model()
        m.obj = Objective(expr=m.b.x[2])
        m.b.deactivate()
        TransformationFactory('core.relax_integer_vars').apply_to(
            m,
            transform_deactivated_blocks=False,
            var_collector=VarCollector.FromExpressions,
        )
        # not transformed
        self.assertIs(m.b.x[1].domain, Integers)
        self.assertEqual(m.b.x[1].lb, 2)
        self.assertEqual(m.b.x[1].ub, 45)
        # transformed
        self.assertIs(m.b.x[2].domain, Reals)
        self.assertEqual(m.b.x[2].lb, 2)
        self.assertEqual(m.b.x[2].ub, 45)
        # not transformed
        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        # not transformed
        self.assertIs(m.y.domain, Binary)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)

    def test_relax_integer_vars_fixed_vars(self):
        m = _make_hierarchical_model()
        m.y.fix(0)
        m.b.y.fix(1)
        reverse = TransformationFactory('core.relax_integer_vars').apply_to(m)

        # change the domain, but don't unfix
        self.assertIs(m.y.domain, Reals)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)
        self.assertEqual(value(m.y), 0)
        self.assertTrue(m.y.fixed)

        self.assertIs(m.b.y.domain, Reals)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        self.assertEqual(value(m.b.y), 1)
        self.assertTrue(m.b.y.fixed)

        # transformed
        for i in [1, 2]:
            self.assertIs(m.b.x[i].domain, Reals)
            self.assertEqual(m.b.x[i].lb, 2)
            self.assertEqual(m.b.x[i].ub, 45)

        # reverse and make sure fixed guys are still fixed
        TransformationFactory('core.relax_integer_vars').apply_to(m, reverse=reverse)
        self.assertIs(m.y.domain, Binary)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, 1)
        self.assertEqual(value(m.y), 0)
        self.assertTrue(m.y.fixed)

        self.assertIs(m.b.y.domain, Binary)
        self.assertEqual(m.b.y.lb, 0)
        self.assertEqual(m.b.y.ub, 1)
        self.assertEqual(value(m.b.y), 1)
        self.assertTrue(m.b.y.fixed)

    @unittest.skipIf(len(solvers) == 0, "LP/MIP solver not available")
    def test_solve_fix_transform(self):
        s = SolverFactory(solvers[0])
        m = _generateModel()
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        m.pprint()
        self.assertEqual(len(m.dual), 0)

        TransformationFactory('core.fix_discrete').apply_to(m)
        self.assertIs(m.x.domain, Binary)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 1)
        s.solve(m)
        self.assertEqual(len(m.dual), 2)
        self.assertAlmostEqual(m.dual[m.c1], -1, 4)
        self.assertAlmostEqual(m.dual[m.c2], 0, 4)


if __name__ == "__main__":
    unittest.main()
