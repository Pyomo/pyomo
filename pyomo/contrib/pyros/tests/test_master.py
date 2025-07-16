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

"""
Test methods for construction and solution of master problem
objects.
"""


import logging
import time
import pyomo.common.unittest as unittest

from pyomo.common.collections import Bunch
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.core.base import ConcreteModel, Constraint, minimize, Objective, Param, Var
from pyomo.core.expr import exp
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import SolverFactory
from pyomo.opt import TerminationCondition

from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.contrib.pyros.master_problem_methods import (
    add_scenario_block_to_master_problem,
    construct_initial_master_problem,
    construct_master_feasibility_problem,
    construct_dr_polishing_problem,
    MasterProblemData,
    higher_order_decision_rule_efficiency,
)
from pyomo.contrib.pyros.util import (
    ModelData,
    preprocess_model_data,
    get_all_first_stage_eq_cons,
    ObjectiveType,
    time_code,
    TimingData,
    VariablePartitioning,
    pyrosTerminationCondition,
)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Packages numpy and scipy must both be available.")

_baron = SolverFactory("baron")
baron_available = _baron.available()
baron_license_is_valid = _baron.license_is_valid()


logger = logging.getLogger(__name__)


def build_simple_model_data(objective_focus="worst_case", decision_rule_order=1):
    """
    Test construction of master problem.
    """
    m = ConcreteModel()
    m.u = Param(initialize=0.5, mutable=True)
    m.x1 = Var(bounds=[-1000, 1000], initialize=1)
    m.x2 = Var(bounds=[-1000, 1000], initialize=1)
    m.x3 = Var(bounds=[-1000, 1000], initialize=-3)
    m.con = Constraint(expr=exp(m.u - 1) - m.x1 - m.x2 * m.u - m.x3 * m.u**2 <= 0)
    m.eq_con = Constraint(expr=m.x2 - 1 == 0)

    m.obj = Objective(expr=m.x1 + m.x2 / 2 + m.x3 / 3)

    config = Bunch(
        uncertain_params=[m.u],
        objective_focus=ObjectiveType[objective_focus],
        decision_rule_order=decision_rule_order,
        progress_logger=logger,
        nominal_uncertain_param_vals=[0.4],
        separation_priority_order=dict(),
        uncertainty_set=BoxSet([[1, 2]]),
    )
    model_data = ModelData(original_model=m, timing=TimingData(), config=config)
    user_var_partitioning = VariablePartitioning(
        first_stage_variables=[m.x1],
        second_stage_variables=[m.x2, m.x3],
        state_variables=[],
    )

    preprocess_model_data(model_data, user_var_partitioning)

    return model_data


class TestConstructMasterProblem(unittest.TestCase):
    """
    Tests for construction of the master problem and
    scenario sub-blocks.
    """

    def test_initial_construct_master(self):
        """
        Test initial construction of the master problem
        from the preprocesed working model.
        """
        model_data = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data)

        self.assertTrue(hasattr(master_model, "scenarios"))
        self.assertIsNot(master_model.scenarios[0, 0], model_data.working_model)
        self.assertTrue(master_model.epigraph_obj.active)
        self.assertIs(
            master_model.epigraph_obj.expr,
            master_model.scenarios[0, 0].first_stage.epigraph_var,
        )

        # check all the variables (including first-stage ones)
        # were cloned
        nadj_var_zip = zip(
            master_model.scenarios[0, 0].all_nonadjustable_variables,
            model_data.working_model.all_nonadjustable_variables,
        )
        for master_var, wm_var in nadj_var_zip:
            self.assertIsNot(
                master_var,
                wm_var,
                f"Variable with name {wm_var.name!r} not cloned as expected.",
            )

        # check parameter value is set to the nominal realization
        self.assertEqual(
            master_model.scenarios[0, 0].user_model.u.value,
            model_data.config.nominal_uncertain_param_vals[0],
        )

    def test_add_scenario_block_to_master(self):
        """
        Test method for adding scenario block to an already
        constructed master problem, without cloning of the
        first-stage variables.
        """
        model_data = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data)
        add_scenario_block_to_master_problem(
            master_model=master_model,
            scenario_idx=[0, 1],
            param_realization=[0.6],
            from_block=master_model.scenarios[0, 0],
            clone_first_stage_components=False,
        )

        self.assertEqual(master_model.scenarios[0, 1].user_model.u.value, 0.6)

        nadj_var_zip = zip(
            master_model.scenarios[0, 0].all_nonadjustable_variables,
            master_model.scenarios[0, 1].all_nonadjustable_variables,
        )
        for var_00, var_01 in nadj_var_zip:
            self.assertIs(
                var_00,
                var_01,
                msg=f"Variable {var_00.name} was cloned across scenario blocks.",
            )

        # the first-stage inequality and equality constraints
        # should be cloned. we do this to avoid issues with the solver
        # interfaces (such as issues with manipulating symbol maps)
        nadj_ineq_con_zip = zip(
            master_model.scenarios[0, 0].first_stage.inequality_cons.values(),
            master_model.scenarios[0, 1].first_stage.inequality_cons.values(),
        )
        for ineq_con_00, ineq_con_01 in nadj_ineq_con_zip:
            self.assertIsNot(
                ineq_con_00,
                ineq_con_01,
                msg=(
                    f"first-stage inequality con {ineq_con_00.name!r} was not "
                    "cloned across scenario blocks."
                ),
            )
            self.assertTrue(
                ineq_con_00.active,
                msg=(
                    "First-stage inequality constraint "
                    f"{ineq_con_00.name!r} should be active."
                ),
            )
            self.assertFalse(
                ineq_con_01.active,
                msg=(
                    "Duplicate first-stage inequality constraint "
                    f"{ineq_con_01.name!r} should be deactivated"
                ),
            )

        nadj_eq_con_zip = zip(
            get_all_first_stage_eq_cons(master_model.scenarios[0, 0]),
            get_all_first_stage_eq_cons(master_model.scenarios[0, 1]),
        )
        for eq_con_00, eq_con_01 in nadj_eq_con_zip:
            self.assertIsNot(
                eq_con_00,
                eq_con_01,
                msg=(
                    f"first-stage equality con {eq_con_00.name} was not cloned "
                    "across scenario blocks."
                ),
            )
            self.assertTrue(
                eq_con_00.active,
                msg=(
                    "First-stage equality constraint "
                    f"{eq_con_00.name!r} should be active."
                ),
            )
            self.assertFalse(
                eq_con_01.active,
                msg=(
                    "Duplicate first-stage equality constraint "
                    f"{eq_con_01.name!r} should be deactivated"
                ),
            )


class TestNewConstructMasterFeasibilityProblem(unittest.TestCase):
    """
    Test construction of the master feasibility problem.
    """

    def build_simple_master_data(self):
        """
        Construct master data-like object for feasibility problem
        tests.
        """
        model_data = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data)
        add_scenario_block_to_master_problem(
            master_model=master_model,
            scenario_idx=[1, 0],
            param_realization=[1],
            from_block=master_model.scenarios[0, 0],
            clone_first_stage_components=False,
        )
        master_data = Bunch(
            master_model=master_model, iteration=1, config=model_data.config
        )

        return master_data

    def test_construct_master_feasibility_problem_var_map(self):
        """
        Test construction of feasibility problem var map.
        """
        master_data = self.build_simple_master_data()
        slack_model = construct_master_feasibility_problem(master_data)

        self.assertTrue(master_data.feasibility_problem_varmap)
        for mvar, feasvar in master_data.feasibility_problem_varmap:
            self.assertIs(
                mvar,
                master_data.master_model.find_component(feasvar),
                msg=f"{mvar.name!r} is not same as find_component({feasvar.name!r})",
            )
            self.assertIs(
                feasvar,
                slack_model.find_component(mvar),
                msg=f"{feasvar.name!r} is not same as find_component({mvar.name!r})",
            )

    def test_construct_master_feasibility_problem_slack_vars(self):
        """
        Check master feasibility slack variables.
        """
        master_data = self.build_simple_master_data()
        slack_model = construct_master_feasibility_problem(master_data)

        slack_var_blk = slack_model._core_add_slack_variables
        scenario_10_blk = slack_model.scenarios[1, 0]

        # test a few of the constraints
        slack_user_model_x3_lb_con = scenario_10_blk.second_stage.inequality_cons[
            "var_x3_certain_lower_bound_con"
        ]
        slack_user_model_x3_lb_con_var = slack_var_blk.find_component(
            "'_slack_minus_scenarios[1,0].second_stage.inequality_cons["
            "var_x3_certain_lower_bound_con]'"
        )
        assertExpressionsEqual(
            self,
            slack_user_model_x3_lb_con.body <= slack_user_model_x3_lb_con.upper,
            -scenario_10_blk.user_model.x3 - slack_user_model_x3_lb_con_var <= 1000.0,
        )
        self.assertEqual(slack_user_model_x3_lb_con_var.value, 0)

        slack_user_model_x3_ub_con = scenario_10_blk.second_stage.inequality_cons[
            "var_x3_certain_upper_bound_con"
        ]
        slack_user_model_x3_ub_con_var = slack_var_blk.find_component(
            "'_slack_minus_scenarios[1,0].second_stage.inequality_cons["
            "var_x3_certain_upper_bound_con]'"
        )
        assertExpressionsEqual(
            self,
            slack_user_model_x3_ub_con.body <= slack_user_model_x3_ub_con.upper,
            scenario_10_blk.user_model.x3 - slack_user_model_x3_ub_con_var <= 1000.0,
        )
        self.assertEqual(slack_user_model_x3_lb_con_var.value, 0)

        # constraint 'con' is violated when u = 0.8;
        # check slack initialization
        slack_user_model_con_var = slack_var_blk.find_component(
            "'_slack_minus_scenarios[1,0].second_stage.inequality_cons"
            "[ineq_con_con_upper_bound_con]'"
        )
        self.assertEqual(
            slack_user_model_con_var.value,
            -master_data.master_model.scenarios[1, 0].user_model.con.uslack(),
        )

    def test_construct_master_feasibility_problem_obj(self):
        """
        Check master feasibility slack variables.
        """
        master_data = self.build_simple_master_data()
        slack_model = construct_master_feasibility_problem(master_data)

        self.assertFalse(slack_model.epigraph_obj.active)
        self.assertTrue(slack_model._core_add_slack_variables._slack_objective.active)


class TestDRPolishingProblem(unittest.TestCase):
    """
    Tests for the PyROS DR polishing problem.
    """

    def build_simple_master_data(self):
        """
        Construct master data-like object for feasibility problem
        tests.
        """
        model_data = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data)
        add_scenario_block_to_master_problem(
            master_model=master_model,
            scenario_idx=[1, 0],
            param_realization=[0.1],
            from_block=master_model.scenarios[0, 0],
            clone_first_stage_components=False,
        )
        master_data = Bunch(
            master_model=master_model, iteration=1, config=model_data.config
        )

        return master_data

    def test_construct_dr_polishing_problem_nonadj_components(self):
        """
        Test state of the nonadjustable components
        of the DR polishing problem.
        """
        master_data = self.build_simple_master_data()
        polishing_model = construct_dr_polishing_problem(master_data)
        eff_first_stage_vars = polishing_model.scenarios[
            0, 0
        ].effective_var_partitioning.first_stage_variables
        for effective_first_stage_var in eff_first_stage_vars:
            self.assertTrue(
                effective_first_stage_var.fixed,
                msg=(
                    "Effective first-stage variable "
                    f"{effective_first_stage_var.name!r} "
                    "not fixed."
                ),
            )

        nom_polishing_block = polishing_model.scenarios[0, 0]
        self.assertTrue(nom_polishing_block.first_stage.epigraph_var.fixed)
        self.assertFalse(nom_polishing_block.first_stage.decision_rule_vars[0][0].fixed)
        self.assertFalse(nom_polishing_block.first_stage.decision_rule_vars[0][1].fixed)

        # ensure constraints in fixed vars were deactivated
        self.assertFalse(nom_polishing_block.user_model.eq_con.active)

        # these have either unfixed DR or adjustable variables,
        # so they should remain active
        # self.assertTrue(nom_polishing_block.user_model.con.active)
        self.assertTrue(
            nom_polishing_block.second_stage.inequality_cons[
                "ineq_con_con_upper_bound_con"
            ].active
        )
        self.assertTrue(nom_polishing_block.second_stage.decision_rule_eqns[0].active)

    def test_construct_dr_polishing_problem_polishing_components(self):
        """
        Test auxiliary Var/Constraint components of the DR polishing
        problem.
        """
        master_data = self.build_simple_master_data()
        # DR order is 1, and x3 is second-stage.
        # to test fixing efficiency, fix the affine DR variable
        decision_rule_vars = master_data.master_model.scenarios[
            0, 0
        ].first_stage.decision_rule_vars
        decision_rule_vars[0][1].fix()
        polishing_model = construct_dr_polishing_problem(master_data)
        nom_polishing_block = polishing_model.scenarios[0, 0]

        self.assertFalse(decision_rule_vars[0][0].fixed)
        self.assertTrue(polishing_model.polishing_vars[0][0].fixed)
        self.assertFalse(polishing_model.polishing_abs_val_lb_con_0[0].active)
        self.assertFalse(polishing_model.polishing_abs_val_ub_con_0[0].active)

        # polishing components for the affine DR term should be
        # fixed/deactivated since the DR variable was fixed
        self.assertTrue(decision_rule_vars[0][1].fixed)
        self.assertTrue(polishing_model.polishing_vars[0][1].fixed)
        self.assertFalse(polishing_model.polishing_abs_val_lb_con_0[1].active)
        self.assertFalse(polishing_model.polishing_abs_val_ub_con_0[1].active)

        # check initialization of polishing vars
        self.assertEqual(
            polishing_model.polishing_vars[0][0].value,
            abs(nom_polishing_block.first_stage.decision_rule_vars[0][0].value),
        )
        self.assertEqual(
            polishing_model.polishing_vars[0][1].value,
            abs(nom_polishing_block.first_stage.decision_rule_vars[0][1].value),
        )

        assertExpressionsEqual(
            self,
            polishing_model.polishing_obj.expr,
            polishing_model.polishing_vars[0][0] + polishing_model.polishing_vars[0][1],
        )
        self.assertEqual(polishing_model.polishing_obj.sense, minimize)

    def test_construct_dr_polishing_problem_objectives(self):
        """
        Test states of the Objective components of the DR
        polishing model.
        """
        master_data = self.build_simple_master_data()
        polishing_model = construct_dr_polishing_problem(master_data)
        self.assertFalse(polishing_model.epigraph_obj.active)
        self.assertTrue(polishing_model.polishing_obj.active)

    def test_construct_dr_polishing_problem_params_zero(self):
        """
        Check that DR polishing fixes/deactivates components
        for DR expression terms where the product of uncertain
        parameters is below tolerance.
        """
        master_data = self.build_simple_master_data()

        # trigger fixing of the corresponding polishing vars
        master_data.master_model.scenarios[0, 0].user_model.u.set_value(1e-10)
        master_data.master_model.scenarios[1, 0].user_model.u.set_value(1e-11)

        polishing_model = construct_dr_polishing_problem(master_data)

        dr_vars = polishing_model.scenarios[0, 0].first_stage.decision_rule_vars

        # since static DR terms should not be polished
        self.assertTrue(polishing_model.polishing_vars[0][0].fixed)
        self.assertFalse(polishing_model.polishing_abs_val_lb_con_0[0].active)
        self.assertFalse(polishing_model.polishing_abs_val_ub_con_0[0].active)

        # affine term should be fixed to 0,
        # since the uncertain param values are small enough.
        # polishing constraints are deactivated since we don't need them
        self.assertTrue(dr_vars[0][1].fixed)
        self.assertEqual(dr_vars[0][1].value, 0)
        self.assertTrue(polishing_model.polishing_vars[0][1].fixed)
        self.assertFalse(polishing_model.polishing_abs_val_lb_con_0[1].active)
        self.assertFalse(polishing_model.polishing_abs_val_ub_con_0[1].active)


class TestHigherOrderDecisionRuleEfficiency(unittest.TestCase):
    """
    Test efficiency for decision rules.
    """

    def test_higher_order_decision_rule_efficiency(self):
        """
        Test higher-order decision rule efficiency.
        """
        model_data = build_simple_model_data(decision_rule_order=2)
        master_model = construct_initial_master_problem(model_data)
        master_data = Bunch(
            master_model=master_model, iteration=0, config=model_data.config
        )
        decision_rule_vars = master_data.master_model.scenarios[
            0, 0
        ].first_stage.decision_rule_vars[0]

        for iter_num in range(4):
            master_data.iteration = iter_num
            higher_order_decision_rule_efficiency(master_data)
            self.assertFalse(
                decision_rule_vars[0].fixed,
                msg=(
                    f"DR Var {decision_rule_vars[1].name!r} should not "
                    f"be fixed by efficiency in iteration {iter_num}"
                ),
            )
            if iter_num == 0:
                self.assertTrue(
                    decision_rule_vars[1].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[1].name!r} should "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )
                self.assertTrue(
                    decision_rule_vars[2].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[2].name!r} should "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )
            elif iter_num <= len(master_data.config.uncertain_params):
                self.assertFalse(
                    decision_rule_vars[1].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[1].name!r} should not "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )
                self.assertTrue(
                    decision_rule_vars[2].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[2].name!r} should "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )
            else:
                self.assertFalse(
                    decision_rule_vars[1].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[1].name!r} should not "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )
                self.assertFalse(
                    decision_rule_vars[2].fixed,
                    msg=(
                        f"DR Var {decision_rule_vars[2].name!r} should not "
                        f"be fixed by efficiency in iteration {iter_num}"
                    ),
                )


class TestSolveMaster(unittest.TestCase):
    """
    Test method for solving master problem
    """

    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_solve_master(self):
        model_data = build_simple_model_data()
        model_data.timing = TimingData()
        baron = SolverFactory("baron")
        model_data.config.update(
            dict(
                local_solver=baron,
                global_solver=baron,
                backup_local_solvers=[],
                backup_global_solvers=[],
                tee=False,
            )
        )
        master_data = MasterProblemData(model_data)
        with time_code(master_data.timing, "main", is_main_timer=True):
            master_soln = master_data.solve_master()
            self.assertEqual(len(master_soln.master_results_list), 1)
            self.assertIsNone(master_soln.feasibility_problem_results)
            self.assertIsNone(master_soln.pyros_termination_condition)
            self.assertIs(master_soln.master_model, master_data.master_model)
            self.assertEqual(
                master_soln.master_results_list[0].solver.termination_condition,
                TerminationCondition.optimal,
                msg=(
                    "Could not solve simple master problem with solve_master "
                    "function."
                ),
            )

    @unittest.skipUnless(baron_available, "Global NLP solver is not available")
    def test_solve_master_timeout_on_master(self):
        """
        Test method for solution of master problems times out
        on feasibility problem.
        """
        model_data = build_simple_model_data()
        model_data.timing = TimingData()
        baron = SolverFactory("baron")
        model_data.config.update(
            dict(
                local_solver=baron,
                global_solver=baron,
                backup_local_solvers=[],
                backup_global_solvers=[],
                tee=False,
                time_limit=1,
            )
        )
        master_data = MasterProblemData(model_data)
        with time_code(master_data.timing, "main", is_main_timer=True):
            time.sleep(1)
            master_soln = master_data.solve_master()
            self.assertIsNone(master_soln.feasibility_problem_results)
            self.assertEqual(master_soln.master_model, master_data.master_model)
            self.assertEqual(len(master_soln.master_results_list), 1)
            self.assertEqual(
                master_soln.master_results_list[0].solver.termination_condition,
                TerminationCondition.optimal,
                msg=(
                    "Could not solve simple master problem with solve_master "
                    "function."
                ),
            )
            self.assertEqual(
                master_soln.pyros_termination_condition,
                pyrosTerminationCondition.time_out,
            )

    @unittest.skipUnless(baron_available, "Global NLP solver is not available")
    def test_solve_master_timeout_on_master_feasibility(self):
        """
        Test method for solution of master problems times out
        on feasibility problem.
        """
        model_data = build_simple_model_data()
        model_data.timing = TimingData()
        baron = SolverFactory("baron")
        model_data.config.update(
            dict(
                local_solver=baron,
                global_solver=baron,
                backup_local_solvers=[],
                backup_global_solvers=[],
                tee=False,
                time_limit=1,
            )
        )
        master_data = MasterProblemData(model_data)
        add_scenario_block_to_master_problem(
            master_data.master_model,
            scenario_idx=[1, 0],
            param_realization=[0.6],
            from_block=master_data.master_model.scenarios[0, 0],
            clone_first_stage_components=False,
        )
        master_data.iteration = 1
        with time_code(master_data.timing, "main", is_main_timer=True):
            time.sleep(1)
            master_soln = master_data.solve_master()
            self.assertIsNotNone(master_soln.feasibility_problem_results)
            self.assertFalse(master_soln.master_results_list)
            self.assertIs(master_soln.master_model, master_data.master_model)
            self.assertEqual(
                master_soln.pyros_termination_condition,
                pyrosTerminationCondition.time_out,
            )


class TestPolishDRVars(unittest.TestCase):
    """
    Test DR polishing subroutine.
    """

    @unittest.skipUnless(
        baron_license_is_valid, "Global NLP solver is not available and licensed."
    )
    def test_polish_dr_vars(self):
        model_data = build_simple_model_data()
        model_data.timing = TimingData()
        baron = SolverFactory("baron")
        model_data.config.update(
            dict(
                local_solver=baron,
                global_solver=baron,
                backup_local_solvers=[],
                backup_global_solvers=[],
                tee=False,
            )
        )
        master_data = MasterProblemData(model_data)
        add_scenario_block_to_master_problem(
            master_data.master_model,
            scenario_idx=[1, 0],
            param_realization=[0.6],
            from_block=master_data.master_model.scenarios[0, 0],
            clone_first_stage_components=False,
        )
        master_data.iteration = 1

        master_data.timing = TimingData()
        with time_code(master_data.timing, "main", is_main_timer=True):
            master_soln = master_data.solve_master()
            self.assertEqual(
                master_soln.master_results_list[0].solver.termination_condition,
                TerminationCondition.optimal,
            )

            results, success = master_data.solve_dr_polishing()
            self.assertEqual(
                results.solver.termination_condition,
                TerminationCondition.optimal,
                msg="Minimize dr norm did not solve to optimality.",
            )
            self.assertTrue(
                success, msg=f"DR polishing success {success}, expected True."
            )


if __name__ == "__main__":
    unittest.main()
