"""Tests for applying basic steps."""
import pyutilib.th as unittest
from pyomo.core import Constraint, Var, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct

from pyutilib.misc import import_file

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'gdp'))


class TestBasicStep(unittest.TestCase):
    """Tests disjunctive basic steps."""

    def test_improper_basic_step(self):
        model_builder = import_file(
            join(exdir, 'two_rxn_lee', 'two_rxn_model.py'))
        m = model_builder.build_model()
        m.basic_step = apply_basic_step([m.reactor_choice, m.max_demand])
        for disj in m.basic_step.disjuncts.values():
            self.assertEqual(
                disj.improper_constraints[1].body.polynomial_degree(), 2)
            self.assertEqual(
                disj.improper_constraints[1].lower, None)
            self.assertEqual(
                disj.improper_constraints[1].upper, 2)
            self.assertEqual(
                len(disj.improper_constraints), 1)
        self.assertFalse(m.max_demand.active)

    def test_improper_basic_step_linear(self):
        model_builder = import_file(
            join(exdir, 'two_rxn_lee', 'two_rxn_model.py'))
        m = model_builder.build_model(use_mccormick=True)
        m.basic_step = apply_basic_step([
            m.reactor_choice, m.max_demand, m.mccormick_1, m.mccormick_2])
        for disj in m.basic_step.disjuncts.values():
            self.assertIs(
                disj.improper_constraints[1].body, m.P)
            self.assertEqual(
                disj.improper_constraints[1].lower, None)
            self.assertEqual(
                disj.improper_constraints[1].upper, 2)
            self.assertEqual(
                disj.improper_constraints[2].body.polynomial_degree(), 1)
            self.assertEqual(
                disj.improper_constraints[2].lower, None)
            self.assertEqual(
                disj.improper_constraints[2].upper, 0)
            self.assertEqual(
                disj.improper_constraints[3].body.polynomial_degree(), 1)
            self.assertEqual(
                disj.improper_constraints[3].lower, None)
            self.assertEqual(
                disj.improper_constraints[3].upper, 0)
            self.assertEqual(
                len(disj.improper_constraints), 3)
        self.assertFalse(m.max_demand.active)
        self.assertFalse(m.mccormick_1.active)
        self.assertFalse(m.mccormick_2.active)

    def check_constraint_body(self, m, constraint, constant):
        self.assertIsNone(constraint.lower)
        self.assertEqual(constraint.upper, 0)
        repn = generate_standard_repn(constraint.body)
        self.assertEqual(repn.constant, constant)
        self.assertEqual(len(repn.linear_vars), 2)
        ct.check_linear_coef(self, repn, m.a, -1)
        ct.check_linear_coef(self, repn, m.x, 1)

    def check_after_improper_basic_step(self, m):
        for disj in m.basic_step.disjuncts.values():
            self.assertEqual(len(disj.improper_constraints), 1)
            cons = disj.improper_constraints[1]
            self.check_constraint_body(m, cons, -1)

    def test_improper_basic_step_simpleConstraint(self):
        m = models.makeTwoTermDisj()
        m.simple = Constraint(expr=m.x <= m.a + 1)

        m.basic_step = apply_basic_step([m.disjunction, m.simple])
        self.check_after_improper_basic_step(m)

        self.assertFalse(m.simple.active)
        self.assertFalse(m.disjunction.active)

    def test_improper_basic_step_constraintData(self):
        m = models.makeTwoTermDisj()
        @m.Constraint([1, 2])
        def indexed(m, i):
            return m.x <= m.a + i

        m.basic_step = apply_basic_step([m.disjunction, m.indexed[1]])
        self.check_after_improper_basic_step(m)
        
        self.assertFalse(m.indexed[1].active)
        self.assertTrue(m.indexed[2].active)
        self.assertFalse(m.disjunction.active)

    def test_improper_basic_step_indexedConstraint(self):
        m = models.makeTwoTermDisj()
        @m.Constraint([1, 2])
        def indexed(m, i):
            return m.x <= m.a + i

        m.basic_step = apply_basic_step([m.disjunction, m.indexed])
        for disj in m.basic_step.disjuncts.values():
            self.assertEqual(len(disj.improper_constraints), 2)
            cons = disj.improper_constraints[1]
            self.check_constraint_body(m, cons, -1)

            cons = disj.improper_constraints[2]
            self.check_constraint_body(m, cons, -2)

    def test_indicator_var_references(self):
        m = models.makeTwoTermDisj()
        m.simple = Constraint(expr=m.x <= m.a + 1)

        m.basic_step = apply_basic_step([m.disjunction, m.simple])

        refs = [v for v in m.basic_step.component_data_objects(
            Var, sort=SortComponents.deterministic)]
        self.assertEqual(len(refs), 2)
        self.assertIs(refs[0][None], m.d[0].indicator_var)
        self.assertIs(refs[1][None], m.d[1].indicator_var)

if __name__ == '__main__':
    unittest.main()
