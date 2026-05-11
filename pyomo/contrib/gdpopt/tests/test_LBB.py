# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests for the Logic-based Branch and Bound solver plugin."""

from io import StringIO
import logging
from math import fabs
from os.path import abspath, dirname, join, normpath

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept
import pyomo.contrib.gdpopt.branch_and_bound as bb
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.branch_and_bound import GDP_LBB_Solver
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_algebraic_variable_list,
    add_util_block,
)
from pyomo.contrib.satsolver.satsolver import z3_available
from pyomo.environ import (
    Binary,
    SolverFactory,
    value,
    ConcreteModel,
    Var,
    Objective,
    maximize,
    minimize,
)
from pyomo.gdp import Disjunction
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', '..', 'examples', 'gdp'))

minlp_solver = 'baron'
minlp_args = dict()
solver_available = SolverFactory(minlp_solver).available()
license_available = (
    SolverFactory(minlp_solver).license_is_valid() if solver_available else False
)


class _RecordingSolver:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def solve(self, model, **kwds):
        has_unfixed_discrete = any(
            (v.is_binary() or v.is_integer()) and not v.fixed
            for v in model.component_data_objects(Var, descend_into=True)
        )
        self.calls.append((self.name, has_unfixed_discrete, kwds))

        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.optimal
        results.problem.lower_bound = 0
        results.problem.upper_bound = 0
        return results


class TestGDPoptLBBNodeSolverDispatch(unittest.TestCase):
    def _make_lbb_solver(self, model):
        solver = GDP_LBB_Solver()
        solver.pyomo_results = SolverResults()
        solver.pyomo_results.problem.sense = minimize
        solver.original_util_block = add_util_block(model)
        add_algebraic_variable_list(solver.original_util_block)
        return solver

    def _make_config(self, solver, relaxed_nlp_solver=None):
        config = solver.CONFIG()
        config.minlp_solver = 'sentinel_minlp'
        config.nlp_solver = 'sentinel_nlp'
        config.local_minlp_solver = 'sentinel_local_minlp'
        config.minlp_solver_args['role'] = 'minlp'
        config.nlp_solver_args['role'] = 'nlp'
        config.local_minlp_solver_args['role'] = 'local_minlp'
        config.relaxed_nlp_solver = relaxed_nlp_solver
        if relaxed_nlp_solver is not None:
            config.relaxed_nlp_solver_args['role'] = 'relaxed_nlp'
        config.integer_tolerance = 1e-5
        config.time_limit = None
        return config

    def _record_solver_calls(self, method_name, model, relaxed_nlp_solver=None):
        calls = []
        original_solver_factory = bb.SolverFactory
        bb.SolverFactory = lambda name: _RecordingSolver(name, calls)
        try:
            solver = self._make_lbb_solver(model)
            config = self._make_config(solver, relaxed_nlp_solver)
            getattr(solver, method_name)(model, config)
        finally:
            bb.SolverFactory = original_solver_factory
        return calls

    def test_continuous_node_subproblem_uses_relaxed_nlp_solver(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)

        calls = self._record_solver_calls(
            '_solve_rnGDP_subproblem', m, relaxed_nlp_solver='sentinel_relaxed_nlp'
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'sentinel_relaxed_nlp')
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[0][2]['role'], 'relaxed_nlp')

    def test_continuous_node_subproblem_defaults_to_minlp_solver(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)

        calls = self._record_solver_calls('_solve_rnGDP_subproblem', m)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'sentinel_minlp')
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[0][2]['role'], 'minlp')

    def test_mixed_integer_node_subproblem_uses_minlp_solver(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.y = Var(domain=Binary)
        m.obj = Objective(expr=m.x + m.y)

        calls = self._record_solver_calls(
            '_solve_rnGDP_subproblem', m, relaxed_nlp_solver='sentinel_relaxed_nlp'
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'sentinel_minlp')
        self.assertTrue(calls[0][1])
        self.assertEqual(calls[0][2]['role'], 'minlp')

    def test_continuous_local_node_subproblem_uses_relaxed_nlp_solver(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.obj = Objective(expr=m.x)

        calls = self._record_solver_calls(
            '_solve_local_rnGDP_subproblem',
            m,
            relaxed_nlp_solver='sentinel_relaxed_nlp',
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'sentinel_relaxed_nlp')
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[0][2]['role'], 'relaxed_nlp')


@unittest.skipUnless(
    solver_available, "Required subsolver %s is not available" % (minlp_solver,)
)
class TestGDPopt_LBB(unittest.TestCase):
    """Tests for logic-based branch and bound."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[[m.x**2 >= 3, m.x >= 3], [m.x**2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        result = SolverFactory('gdpopt.lbb').solve(
            m, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args
        )
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)

    @unittest.skipUnless(z3_available, "Z3 SAT solver is not available")
    def test_infeasible_GDP_check_sat(self):
        """Test for infeasible GDP with check_sat option True."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[[m.x**2 >= 3, m.x >= 3], [m.x**2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            result = SolverFactory('gdpopt.lbb').solve(
                m,
                tee=False,
                check_sat=True,
                minlp_solver=minlp_solver,
                minlp_solver_args=minlp_args,
            )
        self.assertIn(
            "Root node is not satisfiable. Problem is infeasible.",
            output.getvalue().strip(),
        )

        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)

    # This should work--see issue #2483
    # def test_fix_all_but_one_disjunct(self):
    #     m = ConcreteModel()
    #     m.x = Var(bounds=(0, 3))
    #     m.d = Disjunction(expr=[
    #         [m.x ** 2 >= 3, m.x >= 3],
    #         [m.x ** 2 <= -1, m.x <= -1]])
    #     m.o = Objective(expr=m.x)
    #     m.d.disjuncts[1].indicator_var.fix(False)
    #     result = SolverFactory('gdpopt.lbb').solve(
    #         m, tee=True,
    #         minlp_solver=minlp_solver,
    #         minlp_solver_args=minlp_args,
    #     )
    #     self.assertEqual(result.solver.termination_condition,
    #                      TerminationCondition.optimal)
    #     self.assertAlmostEqual(value(m.x), 3)
    #     self.assertTrue(value(m.d.disjuncts[0].indicator_var))
    #     self.assertFalse(value(m.d.disjuncts[1].indicator_var))

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.lbb').solve(
            eight_process,
            tee=False,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        ct.check_8PP_solution(self, eight_process, results)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP_max(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        obj = next(eight_process.component_data_objects(Objective, active=True))
        obj.sense = maximize
        obj.set_value(-1 * obj.expr)
        SolverFactory('gdpopt.lbb').solve(
            eight_process,
            tee=False,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertAlmostEqual(value(eight_process.profit.expr), -68, places=1)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_strip_pack(self):
        """Test logic-based branch and bound with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.lbb').solve(
            strip_pack,
            tee=False,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 1e-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    @unittest.pytest.mark.expensive
    def test_LBB_constrained_layout(self):
        """Test LBB with constrained layout."""
        exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt.lbb').solve(
            cons_layout,
            tee=False,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value,
        )

    def test_LBB_ex_633_trespalacios(self):
        """Test LBB with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt.lbb').solve(
            model, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args
        )
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    @unittest.skipUnless(
        SolverFactory('bonmin').available(exception_flag=False),
        "Bonmin is not available",
    )
    def test_LBB_8PP_with_screening(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.lbb').solve(
            eight_process,
            tee=False,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
            solve_local_rnGDP=True,
            local_minlp_solver='bonmin',
            local_minlp_solver_args={},
        )
        ct.check_8PP_solution(self, eight_process, results)


@unittest.skipUnless(
    solver_available, "Required subsolver %s is not available" % (minlp_solver,)
)
@unittest.skipUnless(z3_available, "Z3 SAT solver is not available.")
class TestGDPopt_LBB_Z3(unittest.TestCase):
    """Tests for logic-based branch and bound with Z3 SAT solver integration."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 2))
        m.d = Disjunction(expr=[[m.x**2 >= 3, m.x >= 3], [m.x**2 <= -1, m.x <= -1]])
        m.o = Objective(expr=m.x)
        result = SolverFactory('gdpopt.lbb').solve(
            m, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args
        )
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_8PP(self):
        """Test the logic-based branch and bound algorithm."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.lbb').solve(
            eight_process,
            tee=False,
            check_sat=True,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        ct.check_8PP_solution(self, eight_process, results)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    def test_LBB_strip_pack(self):
        """Test logic-based branch and bound with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.lbb').solve(
            strip_pack,
            tee=False,
            check_sat=True,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 1e-2)

    @unittest.skipUnless(license_available, "Problem is too big for unlicensed BARON.")
    @unittest.pytest.mark.expensive
    def test_LBB_constrained_layout(self):
        """Test LBB with constrained layout."""
        exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt.lbb').solve(
            cons_layout,
            tee=False,
            check_sat=True,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(
            fabs(objective_value - 41573) <= 200,
            "Objective value of %s instead of 41573" % objective_value,
        )

    def test_LBB_ex_633_trespalacios(self):
        """Test LBB with Francisco thesis example."""
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        model = exfile.build_simple_nonconvex_gdp()
        SolverFactory('gdpopt').solve(
            model,
            algorithm='LBB',
            tee=False,
            check_sat=True,
            minlp_solver=minlp_solver,
            minlp_solver_args=minlp_args,
        )
        objective_value = value(model.obj.expr)
        self.assertAlmostEqual(objective_value, 4.46, 2)


if __name__ == '__main__':
    unittest.main()
