
"""Tests for the GDPopt solver plugin."""
import logging
from math import fabs
from os.path import abspath, dirname, join, normpath

import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, Objective, SolverFactory, Var, value, Integers, Block, Constraint, maximize
from pyomo.gdp import Disjunct, Disjunction
from pyutilib.misc import import_file
from pyomo.opt import TerminationCondition

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'gdp'))

minlp_solver = 'baron'


@unittest.skipIf(not SolverFactory(minlp_solver).available(),
                 "Required subsolver %s is not available"
                 % (minlp_solver,))
class TestGLOA(unittest.TestCase):
    """Tests for global logic-based outer approximation."""

    def test_GLOA_8PP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpbb').solve(
            eight_process, tee=False,
            solver=minlp_solver,
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_GLOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpbb').solve(
            strip_pack, tee=False,
            solver=minlp_solver,
        )
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    def test_GLOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpbb').solve(
            cons_layout, tee=False,
            solver=minlp_solver,
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_GLOA_ex_633_trespalacios(self):
        """Test LOA with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpbb').solve(
            model, tee=False,
            solver=minlp_solver,
        )
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)


if __name__ == '__main__':
    unittest.main()
