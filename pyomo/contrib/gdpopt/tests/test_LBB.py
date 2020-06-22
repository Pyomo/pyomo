"""Tests for the Logic-based Branch and Bound solver plugin."""

from math import fabs
from os.path import abspath, dirname, join, normpath

import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.contrib.satsolver.satsolver import z3_available
from pyomo.environ import SolverFactory, value, ConcreteModel, Var, Objective, maximize
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', '..', 'examples', 'gdp'))

minlp_solver = 'baron'
minlp_args = dict()
solver_available = SolverFactory(minlp_solver).available()
license_available = SolverFactory(minlp_solver).license_is_valid() if solver_available else False


@unittest.skipUnless(solver_available, "Required subsolver %s is not available" % (minlp_solver,))
class TestGDPopt_LBB(unittest.TestCase):
    """Tests for logic-based branch and bound."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        result = SolverFactory('gdpopt').solve(
            m, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertEqual(result.solver.termination_condition, TerminationCondition.infeasible)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP_max(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        obj = next(eight_process.component_data_objects(Objective, active=True))
        obj.sense = maximize
        obj.set_value(-1 * obj.expr)
        SolverFactory('gdpopt').solve(
            eight_process, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertAlmostEqual(value(eight_process.profit.expr), -68, places=1)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_strip_pack(self):
        """Test logic-based branch and bound with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_constrained_layout(self):
        """Test LBB with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_LBB_ex_633_trespalacios(self):
        """Test LBB with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt').solve(
            model, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    @unittest.skipUnless(SolverFactory('bonmin').available(exception_flag=False), "Bonmin is not avaialable")
    def test_LBB_8PP_with_screening(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
            solve_local_rnGDP=True,
            local_minlp_solver='bonmin',
            local_minlp_solver_args={},
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)


@unittest.skipUnless(solver_available, "Required subsolver %s is not available" % (minlp_solver,))
@unittest.skipUnless(z3_available, "Z3 SAT solver is not available.")
class TestGDPopt_LBB_Z3(unittest.TestCase):
    """Tests for logic-based branch and bound with Z3 SAT solver integration."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        result = SolverFactory('gdpopt').solve(
            m, tee=False,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertEqual(result.solver.termination_condition, TerminationCondition.infeasible)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, tee=False, check_sat=True,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_strip_pack(self):
        """Test logic-based branch and bound with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, tee=False, check_sat=True,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_constrained_layout(self):
        """Test LBB with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, tee=False, check_sat=True,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_LBB_ex_633_trespalacios(self):
        """Test LBB with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt').solve(
            model, tee=False, check_sat=True,
            strategy='LBB',
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)


if __name__ == '__main__':
    unittest.main()
