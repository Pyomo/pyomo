"""
Test methods for construction and solution of master problem
objects.
"""


import logging
import unittest

from pyomo.core.base import (
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    Var,
)
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.core.expr import exp

from pyomo.contrib.pyros.util import (
    new_preprocess_model_data,
    VariablePartitioning,
    ObjectiveType,
)
from pyomo.contrib.pyros.master_problem_methods import (
    construct_initial_master_problem,
    add_scenario_block_to_master_problem,
)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Packages numpy and scipy must both be available.")


logger = logging.getLogger(__name__)


def build_simple_model_data(objective_focus="worst_case"):
    """
    Test construction of master problem.
    """
    m = ConcreteModel()
    m.u = Param(initialize=0.5, mutable=True)
    m.x1 = Var(bounds=[-1000, 1000])
    m.x2 = Var(bounds=[-1000, 1000])
    m.x3 = Var(bounds=[-1000, 1000])
    m.con = Constraint(
        expr=exp(m.u - 1) - m.x1 - m.x2 * m.u - m.x3 * m.u**2 <= 0,
    )
    m.eq_con = Constraint(expr=m.x2 - 1 == 0)

    m.obj = Objective(expr=m.x1 + m.x2 / 2 + m.x3 / 3)

    config = Bunch(
        uncertain_params=[m.u],
        objective_focus=ObjectiveType[objective_focus],
        decision_rule_order=1,
        progress_logger=logger,
        nominal_uncertain_param_vals=[0.4],
    )
    model_data = Bunch(original_model=m)
    user_var_partitioning = VariablePartitioning(
        first_stage_variables=[m.x1],
        second_stage_variables=[m.x2, m.x3],
        state_variables=[],
    )

    new_preprocess_model_data(model_data, config, user_var_partitioning)

    return model_data, config


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
        model_data, config = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data, config)

        self.assertTrue(hasattr(master_model, "scenarios"))
        self.assertIsNot(master_model.scenarios[0, 0], model_data.working_model)
        self.assertTrue(master_model.epigraph_obj.active)
        self.assertIs(
            master_model.epigraph_obj.expr,
            master_model.scenarios[0, 0].epigraph_var,
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
            config.nominal_uncertain_param_vals[0],
        )

    def test_add_scenario_block_to_master(self):
        """
        Test method for adding scenario block to an already
        constructed master problem, without cloning of the
        first-stage components.
        """
        model_data, config = build_simple_model_data()
        master_model = construct_initial_master_problem(model_data, config)
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

        nadj_ineq_con_zip = zip(
            master_model.scenarios[0, 0].effective_first_stage_inequality_cons,
            master_model.scenarios[0, 1].effective_first_stage_inequality_cons,
        )
        for ineq_con_00, ineq_con_01 in nadj_ineq_con_zip:
            self.assertIs(
                ineq_con_00,
                ineq_con_01,
                msg=(
                    f"first-stage inequality con {ineq_con_00.name} was cloned "
                    "across scenario blocks."
                ),
            )

        nadj_eq_con_zip = zip(
            master_model.scenarios[0, 0].effective_first_stage_equality_cons,
            master_model.scenarios[0, 1].effective_first_stage_equality_cons,
        )
        for eq_con_00, eq_con_01 in nadj_eq_con_zip:
            self.assertIs(
                eq_con_00,
                eq_con_01,
                msg=(
                    f"first-stage equality con {eq_con_00.name} was cloned "
                    "across scenario blocks."
                ),
            )
