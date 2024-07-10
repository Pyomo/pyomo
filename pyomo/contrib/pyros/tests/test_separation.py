"""
Test separation problem construction methods.
"""


import logging
import unittest

from pyomo.common.collections import Bunch
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.core.base import (
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    Var,
)
from pyomo.core.expr import exp
from pyomo.core.expr.compare import assertExpressionsEqual

from pyomo.contrib.pyros.separation_problem_methods import construct_separation_problem
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.contrib.pyros.util import (
    new_preprocess_model_data,
    ObjectiveType,
    VariablePartitioning,
)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Packages numpy and scipy must both be available.")


logger = logging.getLogger(__name__)


def build_simple_model_data(objective_focus="worst_case"):
    """
    Build simple model data object for master problem construction.
    """
    m = ConcreteModel()
    m.u = Param(initialize=0.5, mutable=True)
    m.u2 = Param(initialize=0, mutable=True)
    m.x1 = Var(bounds=[-1000, 1000])
    m.x2 = Var(bounds=[-1000, 1000])
    m.x3 = Var(bounds=[-1000, 1000])
    m.con = Constraint(
        expr=exp(m.u - 1) - m.x1 - m.x2 * m.u - m.x3 * m.u**2 <= 0,
    )

    # this makes x2 nonadjustable
    m.eq_con = Constraint(expr=m.x2 - 1 == 0)

    m.obj = Objective(expr=m.x1 + m.x2 / 2 + m.x3 / 3 + m.u + m.u2)

    config = Bunch(
        uncertain_params=[m.u, m.u2],
        objective_focus=ObjectiveType[objective_focus],
        decision_rule_order=1,
        progress_logger=logger,
        nominal_uncertain_param_vals=[0.5, 0],
        uncertainty_set=BoxSet([[0, 1], [0, 0]]),
    )
    model_data = Bunch(original_model=m)
    user_var_partitioning = VariablePartitioning(
        first_stage_variables=[m.x1],
        second_stage_variables=[m.x2, m.x3],
        state_variables=[],
    )

    new_preprocess_model_data(model_data, config, user_var_partitioning)

    return model_data, config


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
        model_data, config = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data, config)

        # check nonadjustable components fixed/deactivated
        self.assertTrue(separation_model.user_model.x1.fixed)
        self.assertTrue(separation_model.epigraph_var.fixed)
        for indexed_var in separation_model.decision_rule_vars:
            for dr_var in indexed_var.values():
                self.assertTrue(dr_var.fixed, msg=f"DR var {dr_var.name!r} not fixed")

        # first-stage equality constraints should be inactive
        self.assertFalse(separation_model.user_model.eq_con.active)
        for coeff_con in separation_model.coefficient_matching_conlist.values():
            self.assertFalse(
                coeff_con.active,
                msg=f"Coefficient mathcing constraint {coeff_con.name!r} active."
            )

    def test_construct_separation_problem_perf_ineq_cons(self):
        """
        Check performance inequality constraints are deactivated
        and replaced with objectives, as appropriate.
        """
        model_data, config = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data, config)

        # check performance constraints deactivated
        # check these individually
        self.assertFalse(separation_model.epigraph_con.active)
        self.assertFalse(separation_model.user_model.con.active)

        # check expression of performance cons correct
        # check these individually
        # (i.e. uncertain params have been replaced)
        m = separation_model.user_model
        u1_var = separation_model.uncertainty.uncertain_param_var_list[0]
        u2_var = separation_model.uncertainty.uncertain_param_var_list[1]
        assertExpressionsEqual(
            self,
            separation_model.epigraph_con.expr,
            (
                m.x1 + m.x2 / 2 + m.x3 / 3 + u1_var + u2_var
                - separation_model.epigraph_var <= 0
            ),
        )

        self.assertFalse(separation_model.epigraph_con.active)
        self.assertFalse(m.con.active)
        self.assertFalse(m.var_x3_certain_lower_bound_con.active)
        self.assertFalse(m.var_x3_certain_upper_bound_con.active)

        # check performance con expressions match obj expressions
        # (loop through the con to obj map)
        self.assertEqual(
            len(separation_model.perf_ineq_con_to_obj_map),
            len(separation_model.effective_performance_inequality_cons),
        )
        for perf_con, obj in separation_model.perf_ineq_con_to_obj_map.items():
            assertExpressionsEqual(
                self,
                perf_con.body - perf_con.upper,
                obj.expr,
            )

    def test_construct_separation_problem_perf_eq_and_dr_cons(self):
        """
        Check performance and DR equations are appropriately handled
        by the separation problems.
        """
        # check DR equation is active
        model_data, config = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data, config)

        self.assertTrue(separation_model.decision_rule_eqns[0].active)

        u1_var = separation_model.uncertainty.uncertain_param_var_list[0]
        u2_var = separation_model.uncertainty.uncertain_param_var_list[1]
        assertExpressionsEqual(
            self,
            separation_model.decision_rule_eqns[0].expr,
            (
                separation_model.decision_rule_vars[0][0]
                + u1_var * separation_model.decision_rule_vars[0][1]
                + u2_var * separation_model.decision_rule_vars[0][2]
                - separation_model.user_model.x3
                == 0
            ),
        )

    def test_construct_separation_problem_uncertain_param_components(self):
        """
        Test separation problem handles uncertain parameter variable
        components as expected.
        """
        model_data, config = build_simple_model_data(objective_focus="worst_case")
        separation_model = construct_separation_problem(model_data, config)

        # u, bounds [0, 1]
        self.assertFalse(separation_model.uncertainty.uncertain_param_var_list[0].fixed)
        # u2, bounds [0, 0]
        separation_model.uncertainty.uncertain_param_var_list[1].pprint()
        self.assertTrue(separation_model.uncertainty.uncertain_param_var_list[1].fixed)
        for con in separation_model.uncertainty.uncertainty_cons_list:
            self.assertTrue(con.active, f"Uncertainty set con {con.name!r} inactive.")


if __name__ == "__main__":
    unittest.main()
