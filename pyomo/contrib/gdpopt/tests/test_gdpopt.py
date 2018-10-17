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
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', '..', 'examples', 'gdp'))

mip_solver = 'glpk'
nlp_solver = 'ipopt'
global_nlp_solver = 'baron'
global_nlp_solver_args = dict()
LOA_solvers = (mip_solver, nlp_solver)
GLOA_solvers = (mip_solver, global_nlp_solver)
LOA_solvers_available = all(SolverFactory(s).available() for s in LOA_solvers)
GLOA_solvers_available = all(SolverFactory(s).available() for s in GLOA_solvers)


@unittest.skipIf(not LOA_solvers_available,
                 "Required subsolvers %s are not available"
                 % (LOA_solvers,))
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
                m, strategy='LOA',
                mip_solver=mip_solver,
                nlp_solver=nlp_solver)
            self.assertIn("Set covering problem was infeasible.",
                          output.getvalue().strip())

    def test_LOA_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='LOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            tee=False)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='LOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    def test_LOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, strategy='LOA',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
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
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_LOA_strip_pack_maxBinary(self):
        """Test LOA with strip packing using max_binary initialization."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='LOA', init_strategy='max_binary',
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)
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
            mip_solver=mip_solver,
            nlp_solver=nlp_solver)

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
            mip_solver=mip_solver,
            nlp_solver=nlp_solver,
            call_after_subproblem_feasible=assert_correct_disjuncts_active)

        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)


@unittest.skipIf(not GLOA_solvers_available,
                 "Required subsolvers %s are not available"
                 % (GLOA_solvers,))
@unittest.skipIf(not mcpp_available(), "MC++ is not available")
class TestGLOA(unittest.TestCase):
    """Tests for global logic-based outer approximation."""

    def test_GLOA_8PP(self):
        """Test the global logic-based outer approximation algorithm."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        SolverFactory('gdpopt').solve(
            eight_process, strategy='GLOA', tee=False,
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args
        )
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 1E-2)

    def test_GLOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt').solve(
            strip_pack, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args)
        self.assertTrue(
            fabs(value(strip_pack.total_length.expr) - 11) <= 1E-2)

    def test_GLOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt').solve(
            cons_layout, strategy='GLOA',
            mip_solver=mip_solver,
            iterlim=36,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            tee=False)
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value)

    def test_GLOA_ex_633_trespalacios(self):
        """Test LOA with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt').solve(
            model, strategy='GLOA',
            mip_solver=mip_solver,
            nlp_solver=global_nlp_solver,
            nlp_solver_args=global_nlp_solver_args,
            tee=False)
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)



if __name__ == '__main__':
    unittest.main()
