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
Test separation problem construction methods.
"""


import logging
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept

from pyomo.common.collections import Bunch
from pyomo.common.dependencies import numpy as np, numpy_available, scipy_available
from pyomo.core.base import ConcreteModel, Constraint, Objective, Param, Var
from pyomo.core.expr import exp, RangedExpression, value
from pyomo.core.expr.compare import assertExpressionsEqual

from pyomo.contrib.pyros.master_problem_methods import (
    MasterProblemData,
    add_scenario_block_to_master_problem,
)
from pyomo.contrib.pyros.separation_problem_methods import (
    construct_separation_problem,
    group_ss_ineq_constraints_by_priority,
    SeparationProblemData,
    initialize_separation,
)
from pyomo.contrib.pyros.uncertainty_sets import (
    BoxSet,
    FactorModelSet,
    DiscreteScenarioSet,
)
from pyomo.contrib.pyros.util import (
    ModelData,
    preprocess_model_data,
    ObjectiveType,
    VariablePartitioning,
)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Packages numpy and scipy must both be available.")


logger = logging.getLogger(__name__)


def build_simple_model_data(objective_focus="worst_case", uncertainty_set=None):
    """
    Build simple model data object for master problem construction.
    """
    m = ConcreteModel()
    m.u = Param(initialize=0.5, mutable=True)
    m.u2 = Param(initialize=0, mutable=True)
    m.x1 = Var(bounds=[-1000, 1000])
    m.x2 = Var(bounds=[-1000, 1000])
    m.x3 = Var(bounds=[-1000, 1000])
    m.con = Constraint(expr=exp(m.u - 1) - m.x1 - m.x2 * m.u - m.x3 * m.u**2 <= 0)

    # this makes x2 nonadjustable
    m.eq_con = Constraint(expr=m.x2 - 1 == 0)

    m.obj = Objective(expr=m.x1 + m.x2 / 2 + m.x3 / 3 + m.u + m.u2)

    if uncertainty_set is None:
        uncertainty_set = BoxSet([[0, 1], [0, 0]])

    config = Bunch(
        uncertain_params=[m.u, m.u2],
        objective_focus=ObjectiveType[objective_focus],
        decision_rule_order=1,
        progress_logger=logger,
        nominal_uncertain_param_vals=[0.5, 0],
        uncertainty_set=uncertainty_set,
        separation_priority_order=dict(con=2),
    )
    model_data = ModelData(original_model=m, timing=None, config=config)
    user_var_partitioning = VariablePartitioning(
        first_stage_variables=[m.x1],
        second_stage_variables=[m.x2, m.x3],
        state_variables=[],
    )

    preprocess_model_data(model_data, user_var_partitioning)

    return model_data


class TestConstructSeparationProblem(unittest.TestCase):
    """
    Test method for construction of separation problem.
    """

    def test_construct_separation_problem_nonadj_components(self):
        """
        Check first-stage variables and constraints of the
        separation problem are fixed and deactivated,
        respectively.
        """
        model_data = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data)

        # check nonadjustable components fixed/deactivated
        self.assertTrue(separation_model.user_model.x1.fixed)
        self.assertTrue(separation_model.first_stage.epigraph_var.fixed)
        for indexed_var in separation_model.first_stage.decision_rule_vars:
            for dr_var in indexed_var.values():
                self.assertTrue(dr_var.fixed, msg=f"DR var {dr_var.name!r} not fixed")

        # first-stage equality constraints should be inactive
        self.assertFalse(separation_model.user_model.eq_con.active)
        for coeff_con in separation_model.first_stage.coefficient_matching_cons:
            self.assertFalse(
                coeff_con.active,
                msg=f"Coefficient matching constraint {coeff_con.name!r} active.",
            )

    def test_construct_separation_problem_ss_ineq_cons(self):
        """
        Check second-stage inequality constraints are deactivated
        and replaced with objectives, as appropriate.
        """
        model_data = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data)

        # check expression of second-stage ineq cons correct
        # check these individually
        # (i.e. uncertain params have been replaced)
        m = separation_model.user_model
        u1_var = separation_model.uncertainty.uncertain_param_var_list[0]
        u2_var = separation_model.uncertainty.uncertain_param_var_list[1]
        assertExpressionsEqual(
            self,
            separation_model.second_stage.inequality_cons["epigraph_con"].expr,
            (
                m.x1
                + m.x2 / 2
                + m.x3 / 3
                + u1_var
                + u2_var
                - separation_model.first_stage.epigraph_var
                <= 0
            ),
        )

        self.assertFalse(
            separation_model.second_stage.inequality_cons["epigraph_con"].active
        )
        self.assertFalse(
            m.con.active,
            separation_model.second_stage.inequality_cons[
                "ineq_con_con_upper_bound_con"
            ].active,
        )
        self.assertFalse(
            m.con.active,
            separation_model.second_stage.inequality_cons[
                "var_x3_certain_lower_bound_con"
            ].active,
        )
        self.assertFalse(
            m.con.active,
            separation_model.second_stage.inequality_cons[
                "var_x3_certain_upper_bound_con"
            ].active,
        )

        # check second-stage ineq con expressions match obj expressions
        # (loop through the con to obj map)
        self.assertEqual(
            len(separation_model.second_stage_ineq_con_to_obj_map),
            len(separation_model.second_stage.inequality_cons),
        )
        for ineq_con, obj in separation_model.second_stage_ineq_con_to_obj_map.items():
            assertExpressionsEqual(self, ineq_con.body - ineq_con.upper, obj.expr)

    def test_construct_separation_problem_ss_eq_and_dr_cons(self):
        """
        Check second-stage and DR equations are appropriately handled
        by the separation problems.
        """
        # check DR equation is active
        model_data = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data)

        self.assertTrue(separation_model.second_stage.decision_rule_eqns[0].active)

        u1_var = separation_model.uncertainty.uncertain_param_var_list[0]
        assertExpressionsEqual(
            self,
            separation_model.second_stage.decision_rule_eqns[0].expr,
            (
                separation_model.first_stage.decision_rule_vars[0][0]
                # only one effective uncertain parameter,
                # so only one affine DR term
                + u1_var * separation_model.first_stage.decision_rule_vars[0][1]
                - separation_model.user_model.x3
                == 0
            ),
        )

    def test_construct_separation_problem_uncertainty_components(self):
        """
        Test separation problem handles uncertain parameter variable
        components as expected.
        """
        model_data = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data)
        uncertainty_blk = separation_model.uncertainty
        boxcon1, boxcon2 = uncertainty_blk.uncertainty_cons_list
        paramvar1, paramvar2 = uncertainty_blk.uncertain_param_var_list

        self.assertEqual(uncertainty_blk.auxiliary_var_list, [])
        self.assertEqual(len(uncertainty_blk.uncertainty_cons_list), 2)
        assertExpressionsEqual(
            self,
            boxcon1.expr,
            RangedExpression((np.int_(0), paramvar1, np.int_(1)), False),
        )
        assertExpressionsEqual(
            self,
            boxcon2.expr,
            RangedExpression((np.int_(0), paramvar2, np.int_(0)), False),
        )
        self.assertTrue(boxcon1.active)
        # constraints in fixed uncertain params should be deactivated
        self.assertFalse(boxcon2.active)

        # u, bounds [0, 1]
        self.assertFalse(paramvar1.fixed)
        # bounds [0, 0]; separation constructor should fix the Var
        self.assertTrue(paramvar2.fixed)

        self.assertEqual(paramvar1.bounds, (0, 1))
        self.assertEqual(paramvar2.bounds, (0, 0))

    def test_construct_separation_problem_uncertain_factor_param_components(self):
        """
        Test separation problem uncertainty components for uncertainty
        set requiring auxiliary variables.
        """
        model_data = build_simple_model_data(
            objective_focus="worst_case",
            uncertainty_set=FactorModelSet(
                origin=[1, 0], beta=1, number_of_factors=2, psi_mat=[[1, 2.5], [0, 1]]
            ),
        )
        separation_model = construct_separation_problem(model_data)
        uncertainty_blk = separation_model.uncertainty
        *matrix_product_cons, aux_sum_con = uncertainty_blk.uncertainty_cons_list
        paramvar1, paramvar2 = uncertainty_blk.uncertain_param_var_list
        auxvar1, auxvar2 = uncertainty_blk.auxiliary_var_list

        self.assertEqual(len(matrix_product_cons), 2)
        self.assertTrue(matrix_product_cons[0].active)
        self.assertTrue(matrix_product_cons[1].active)
        self.assertTrue(aux_sum_con.active)
        assertExpressionsEqual(
            self, aux_sum_con.expr, RangedExpression((-2, auxvar1 + auxvar2, 2), False)
        )
        assertExpressionsEqual(
            self, matrix_product_cons[0].expr, auxvar1 + 2.5 * auxvar2 + 1 == paramvar1
        )
        assertExpressionsEqual(
            self, matrix_product_cons[1].expr, 0.0 * auxvar1 + auxvar2 == paramvar2
        )

        # none of the vars should be fixed
        self.assertFalse(paramvar1.fixed)
        self.assertFalse(paramvar2.fixed)
        self.assertFalse(auxvar1.fixed)
        self.assertFalse(auxvar2.fixed)

        # factor set auxiliary variables
        self.assertEqual(auxvar1.bounds, (-1, 1))
        self.assertEqual(auxvar2.bounds, (-1, 1))

        # factor set bounds are tighter
        self.assertEqual(paramvar1.bounds, (-2.5, 4.5))
        self.assertEqual(paramvar2.bounds, (-1.0, 1.0))

        # check decision rule equation
        assertExpressionsEqual(
            self,
            separation_model.second_stage.decision_rule_eqns[0].expr,
            (
                separation_model.first_stage.decision_rule_vars[0][0]
                + paramvar1 * separation_model.first_stage.decision_rule_vars[0][1]
                + paramvar2 * separation_model.first_stage.decision_rule_vars[0][2]
                - separation_model.user_model.x3
                == 0
            ),
        )


class TestGroupSecondStageIneqConsByPriority(unittest.TestCase):
    def test_group_ss_ineq_constraints_by_priority(self):
        model_data = build_simple_model_data()
        model_data.config.progress_logger = logging.getLogger(__name__)
        separation_model = construct_separation_problem(model_data)

        # build mock separation data-like object
        # since we are testing only the grouping method
        separation_data = Bunch(
            separation_model=separation_model,
            separation_priority_order=model_data.separation_priority_order,
            config=model_data.config,
        )

        priority_groups = group_ss_ineq_constraints_by_priority(separation_data)

        self.assertEqual(list(priority_groups.keys()), [2, 0])
        ss_ineq_cons = separation_model.second_stage.inequality_cons
        self.assertEqual(
            priority_groups[2], [ss_ineq_cons["ineq_con_con_upper_bound_con"]]
        )
        self.assertEqual(
            priority_groups[0],
            [
                ss_ineq_cons["var_x3_certain_lower_bound_con"],
                ss_ineq_cons["var_x3_certain_upper_bound_con"],
                ss_ineq_cons["epigraph_con"],
            ],
        )


class TestInitializeSeparation(unittest.TestCase):
    """
    Tests for separation subproblem initialization.
    """

    def test_initialize_separation(self):
        model_data = build_simple_model_data(
            objective_focus="worst_case",
            uncertainty_set=FactorModelSet(
                # note: origin is chosen to be different
                #       from the nominal value so that
                #       initialization of auxiliary uncertain
                #       parameters is meaningfully tested
                origin=[1, 0.5],
                psi_mat=[[1, 1], [1, 0]],
                beta=0.5,
                number_of_factors=2,
            ),
        )

        master_data = MasterProblemData(model_data)
        nom_scenario_blk = master_data.master_model.scenarios[0, 0]
        nom_scenario_blk.user_model.x1.set_value(10)
        nom_scenario_blk.user_model.x2.set_value(1)
        nom_scenario_blk.user_model.x3.set_value(5)
        nom_scenario_blk.first_stage.decision_rule_vars[0][0].set_value(5)
        nom_scenario_blk.first_stage.epigraph_var.set_value(
            value(nom_scenario_blk.full_objective)
        )

        # set up new scenario block
        new_master_param_realization = [1.5, 1]
        add_scenario_block_to_master_problem(
            master_model=master_data.master_model,
            scenario_idx=(1, 0),
            param_realization=new_master_param_realization,
            from_block=nom_scenario_blk,
            clone_first_stage_components=False,
        )
        new_scenario_blk = master_data.master_model.scenarios[1, 0]
        new_scenario_blk.first_stage.epigraph_var.set_value(
            # objective for new block is higher,
            # so update the epigraph variable
            value(new_scenario_blk.full_objective)
        )
        # different value for the adjustable variable
        # so we also adjust the DR variables
        # to ensure the DR equations are satisfied
        new_scenario_blk.user_model.x3.set_value(6)
        new_scenario_blk.first_stage.decision_rule_vars[0][0].set_value(4.5)
        new_scenario_blk.first_stage.decision_rule_vars[0][1].set_value(1)

        separation_data = SeparationProblemData(model_data)

        ss_ineq_con_to_maximize = (
            separation_data.separation_model.second_stage.inequality_cons[
                "epigraph_con"
            ]
        )
        separation_data.points_added_to_master[1, 0] = new_master_param_realization
        separation_data.auxiliary_values_for_master_points[(1, 0)] = (
            model_data.config.uncertainty_set.compute_auxiliary_uncertain_param_vals(
                point=new_master_param_realization, solver=None
            )
        )

        with LoggingIntercept(module=__name__, level=logging.DEBUG) as LOG:
            initialize_separation(
                ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
                separation_data=separation_data,
                master_data=master_data,
            )

        # the constraint violation being maximized for
        # has a higher value in the newer master block
        init_from_scenario_blk = new_scenario_blk

        # all active constraints of the separation problem
        # should be satisfied post initialization, so expect
        # no logging messages stating otherwise
        log_output = LOG.getvalue()
        self.assertFalse(log_output, "DEBUG-level log output should be empty.")

        sep_usr_blk = separation_data.separation_model.user_model
        sep_model = separation_data.separation_model

        # first-stage variables added by PyROS
        self.assertEqual(
            value(sep_model.first_stage.epigraph_var),
            value(init_from_scenario_blk.first_stage.epigraph_var),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][0]),
            value(init_from_scenario_blk.first_stage.decision_rule_vars[0][0]),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][1]),
            value(init_from_scenario_blk.first_stage.decision_rule_vars[0][1]),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][2]),
            value(init_from_scenario_blk.first_stage.decision_rule_vars[0][2]),
        )

        # variables of original user model
        self.assertEqual(
            value(sep_usr_blk.x1), value(init_from_scenario_blk.user_model.x1)
        )
        self.assertEqual(
            value(sep_usr_blk.x2), value(init_from_scenario_blk.user_model.x2)
        )
        self.assertEqual(
            value(sep_usr_blk.x3), value(init_from_scenario_blk.user_model.x3)
        )

        # main uncertain parameter variables
        self.assertEqual(
            value(sep_model.uncertainty.uncertain_param_indexed_var[0]),
            value(init_from_scenario_blk.uncertain_params[0]),
        )
        self.assertEqual(
            value(sep_model.uncertainty.uncertain_param_indexed_var[1]),
            value(init_from_scenario_blk.uncertain_params[1]),
        )

        # auxiliary uncertain parameter variables
        expected_aux_var_vals = separation_data.auxiliary_values_for_master_points[
            (1, 0)
        ]
        self.assertEqual(
            value(sep_model.uncertainty.auxiliary_var_list[0]), expected_aux_var_vals[0]
        )
        self.assertEqual(
            value(sep_model.uncertainty.auxiliary_var_list[1]), expected_aux_var_vals[1]
        )

    def test_initialize_separation_infeasibility_logging(self):
        """
        Test initialization of a separation problem for which
        one of the active separation model constraints is
        not satisfied.
        """
        model_data = build_simple_model_data(
            objective_focus="worst_case",
            uncertainty_set=FactorModelSet(
                # note: origin is chosen to be different
                #       from the nominal value so that
                #       initialization of auxiliary uncertain
                #       parameters is meaningfully tested
                origin=[1, 0.5],
                psi_mat=[[1, 1], [1, 0]],
                beta=0.5,
                number_of_factors=2,
            ),
        )

        master_data = MasterProblemData(model_data)
        nom_scenario_blk = master_data.master_model.scenarios[0, 0]
        nom_scenario_blk.user_model.x1.set_value(10)
        nom_scenario_blk.user_model.x2.set_value(1)

        # this results in a violation of the DR equality constraint
        nom_scenario_blk.user_model.x3.set_value(5 + 1.5e-5)
        nom_scenario_blk.first_stage.decision_rule_vars[0][0].set_value(5)

        nom_scenario_blk.first_stage.epigraph_var.set_value(
            value(nom_scenario_blk.full_objective)
        )

        separation_data = SeparationProblemData(model_data)
        ss_ineq_con_to_maximize = (
            separation_data.separation_model.second_stage.inequality_cons[
                "epigraph_con"
            ]
        )
        with LoggingIntercept(module=__name__, level=logging.DEBUG) as LOG:
            initialize_separation(
                ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
                separation_data=separation_data,
                master_data=master_data,
            )
        log_output = LOG.getvalue()
        log_output_lines = log_output.split("\n")[:-1]
        self.assertEqual(
            len(log_output_lines),
            1,
            "Expected DEBUG-level output for separation problem initialization "
            "test to have exactly 1 line.",
        )
        self.assertRegex(
            log_output,
            r"Initial point for separation of .*violates the model constraint "
            r"'second_stage.decision_rule_eqns\[0\].*' by more than.*",
        )

    def test_initialize_separation_trivially_infeas_uncertainty(self):
        """
        Test error raised by separation problem initialization routine
        if the routine finds violated constraints depending
        only on fixed uncertain parameters.
        """
        model_data = build_simple_model_data(
            objective_focus="worst_case",
            # notice: the bounds mean the first uncertain
            # parameter will be fixed
            uncertainty_set=BoxSet(bounds=[[0.5, 0.5], [0, 1]]),
        )

        master_data = MasterProblemData(model_data)
        nom_scenario_blk = master_data.master_model.scenarios[0, 0]
        nom_scenario_blk.user_model.x1.set_value(10)
        nom_scenario_blk.user_model.x2.set_value(1)
        nom_scenario_blk.user_model.x3.set_value(5)
        nom_scenario_blk.first_stage.decision_rule_vars[0][0].set_value(5)
        nom_scenario_blk.first_stage.epigraph_var.set_value(
            value(nom_scenario_blk.full_objective)
        )

        separation_data = SeparationProblemData(model_data)
        # notice: the artificial point is such that the box constraint
        #         in the effectively certain parameter will be violated;
        #         this should trigger an exception
        separation_data.points_added_to_master[(0, 0)] = (1.5, 0)
        ss_ineq_con_to_maximize = (
            separation_data.separation_model.second_stage.inequality_cons[
                "epigraph_con"
            ]
        )
        exc_str = "Trivial infeasibility detected in the.*BoxSet.*constraints"
        with self.assertRaisesRegex(ValueError, exc_str):
            initialize_separation(
                ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
                separation_data=separation_data,
                master_data=master_data,
            )

    def test_initialize_separation_discrete_uncertainty(self):
        """
        Test initialization of a separation problem
        with a discrete uncertainty set.
        """
        model_data = build_simple_model_data(
            objective_focus="worst_case",
            uncertainty_set=DiscreteScenarioSet(scenarios=[[0.5, 0], [1, 0.5]]),
        )

        master_data = MasterProblemData(model_data)
        nom_scenario_blk = master_data.master_model.scenarios[0, 0]
        nom_scenario_blk.user_model.x1.set_value(10)
        nom_scenario_blk.user_model.x2.set_value(1)

        # this results in a violation of the DR equality constraint,
        # which is not logged, since the uncertainty set is discrete
        nom_scenario_blk.user_model.x3.set_value(5 + 1.1e-5)
        nom_scenario_blk.first_stage.decision_rule_vars[0][0].set_value(5)

        nom_scenario_blk.first_stage.epigraph_var.set_value(
            value(nom_scenario_blk.full_objective)
        )

        separation_data = SeparationProblemData(model_data)
        ss_ineq_con_to_maximize = (
            separation_data.separation_model.second_stage.inequality_cons[
                "epigraph_con"
            ]
        )

        sep_usr_blk = separation_data.separation_model.user_model
        sep_model = separation_data.separation_model

        # fix uncertain parameters to off-nominal value;
        # these should not be modified by the initialization
        off_nominal_scenario = model_data.config.uncertainty_set.scenarios[1]
        sep_model.uncertainty.uncertain_param_var_list[0].set_value(
            off_nominal_scenario[0]
        )
        sep_model.uncertainty.uncertain_param_var_list[1].set_value(
            off_nominal_scenario[1]
        )
        sep_model.uncertainty.uncertain_param_var_list[0].fix()
        sep_model.uncertainty.uncertain_param_var_list[1].fix()

        with LoggingIntercept(module=__name__, level=logging.DEBUG) as LOG:
            initialize_separation(
                ss_ineq_con_to_maximize=ss_ineq_con_to_maximize,
                separation_data=separation_data,
                master_data=master_data,
            )
        log_output = LOG.getvalue()
        self.assertFalse(log_output, "DEBUG-level log output should be empty.")

        # first-stage variables added by PyROS
        self.assertEqual(
            value(sep_model.first_stage.epigraph_var),
            value(nom_scenario_blk.first_stage.epigraph_var),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][0]),
            value(nom_scenario_blk.first_stage.decision_rule_vars[0][0]),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][1]),
            value(nom_scenario_blk.first_stage.decision_rule_vars[0][1]),
        )
        self.assertEqual(
            value(sep_model.first_stage.decision_rule_vars[0][2]),
            value(nom_scenario_blk.first_stage.decision_rule_vars[0][2]),
        )

        # variables of original user model
        self.assertEqual(value(sep_usr_blk.x1), value(nom_scenario_blk.user_model.x1))
        self.assertEqual(value(sep_usr_blk.x2), value(nom_scenario_blk.user_model.x2))
        self.assertEqual(value(sep_usr_blk.x3), value(nom_scenario_blk.user_model.x3))

        # uncertain parameter variable state should not have been
        # modified by the initialization
        self.assertEqual(
            value(sep_model.uncertainty.uncertain_param_indexed_var[0]),
            off_nominal_scenario[0],
        )
        self.assertEqual(
            value(sep_model.uncertainty.uncertain_param_indexed_var[1]),
            off_nominal_scenario[1],
        )
        self.assertTrue(sep_model.uncertainty.uncertain_param_var_list[0].fixed)
        self.assertTrue(sep_model.uncertainty.uncertain_param_var_list[1].fixed)


if __name__ == "__main__":
    unittest.main()
