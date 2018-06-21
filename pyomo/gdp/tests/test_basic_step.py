"""Tests for applying basic steps."""
import pyutilib.th as unittest
from pyomo.gdp.basic_step import apply_basic_step

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


if __name__ == '__main__':
    unittest.main()
