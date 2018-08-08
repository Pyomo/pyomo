"""Tests for the GDPopt solver plugin."""
import logging
from math import fabs
from os.path import abspath, dirname, join, normpath

from six import StringIO

import pyomo.core.base.symbolic
import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Objective, SolverFactory, Var, value
from pyomo.gdp import Disjunct, Disjunction
from pyutilib.misc import import_file

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', '..', 'examples', 'gdp'))

required_solvers = ('ipopt', 'glpk')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 "Required subsolvers %s are not available"
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 "Symbolic differentiation is not available")
class TestGDPopt(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[
            [m.x ** 2 >= 3, m.x >= 3],
            [m.x ** 2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            SolverFactory('gdpopt').solve(
                m, strategy='LOA', mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0])
            self.assertIn("Set covering problem was infeasible.",
                          output.getvalue().strip())

    def test_LOA_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='LOA',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='LOA',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    def test_LOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, strategy='LOA',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_LOA_8PP_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='LOA', init_strategy='max_binary',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_maxBinary(self):
        """Test LOA with strip packing using max_binary initialization."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='LOA', init_strategy='max_binary',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    def test_LOA_8PP_fixed_disjuncts(self):
        """Test LOA with 8PP using fixed disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            eight_process.use_unit_1or2.disjuncts[0],
            eight_process.use_unit_3ornot.disjuncts[1],
            eight_process.use_unit_4or5ornot.disjuncts[0],
            eight_process.use_unit_6or7ornot.disjuncts[1],
            eight_process.use_unit_8ornot.disjuncts[0]
        ]
        for disj in eight_process.component_data_objects(Disjunct):
            if disj in initialize:
                disj.indicator_var.set_value(1)
            else:
                disj.indicator_var.set_value(0)
        SolverFactory('gdpopt').solve(
            eight_process, strategy='LOA', init_strategy='fix_disjuncts',
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0])

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [
            # Use units 1, 4, 7, 8
            [eight_process.use_unit_1or2.disjuncts[0],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[1],
             eight_process.use_unit_8ornot.disjuncts[0]],
            # Use units 2, 4, 6, 8
            [eight_process.use_unit_1or2.disjuncts[1],
             eight_process.use_unit_3ornot.disjuncts[1],
             eight_process.use_unit_4or5ornot.disjuncts[0],
             eight_process.use_unit_6or7ornot.disjuncts[0],
             eight_process.use_unit_8ornot.disjuncts[0]]
        ]

        def assert_correct_disjuncts_active(nlp_model, solve_data):
            if solve_data.master_iteration >= 1:
                return  # only checking initialization
            iter_num = solve_data.nlp_iteration
            disjs_should_be_active = initialize[iter_num - 1]
            for orig_disj, soln_disj in zip(
                solve_data.original_model.GDPopt_utils.orig_disjuncts_list,
                nlp_model.GDPopt_utils.orig_disjuncts_list
            ):
                if orig_disj in disjs_should_be_active:
                    self.assertTrue(soln_disj.indicator_var.value == 1)

        SolverFactory('gdpopt').solve(
            eight_process, strategy='LOA', init_strategy='custom_disjuncts',
            custom_init_disjuncts=initialize,
            mip_solver=required_solvers[1],
            nlp_solver=required_solvers[0],
            call_after_subproblem_feasible=assert_correct_disjuncts_active)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)


if __name__ == '__main__':
    unittest.main()
