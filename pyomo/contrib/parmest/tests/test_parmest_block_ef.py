#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026 National Technology and Engineering Solutions of
#  Sandia, LLC Under the terms of Contract DE-NA0003525 with National
#  Technology and Engineering Solutions of Sandia, LLC, the U.S. Government
#  retains certain rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import pandas as pd

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

ipopt_available = pyo.SolverFactory("ipopt").available()


class LinearThetaExperiment(Experiment):
    def __init__(self, x, y, include_second_output=False):
        self.x_data = x
        self.y_data = y
        self.include_second_output = include_second_output
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta = pyo.Var(initialize=0.0, bounds=(-10.0, 10.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))
        m.y_link = pyo.Constraint(expr=m.y == m.theta + m.x)
        if self.include_second_output:
            m.z = pyo.Var(initialize=2.0 * self.y_data)
            m.z_link = pyo.Constraint(expr=m.z == 2.0 * m.theta + m.x)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])
        if self.include_second_output:
            m.experiment_outputs.update([(m.z, float(2.0 * self.y_data))])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])
        if self.include_second_output:
            m.measurement_error.update([(m.z, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def _build_estimator(data, include_second_output=False):
    exp_list = [
        LinearThetaExperiment(x=x, y=y, include_second_output=include_second_output)
        for x, y in data
    ]
    return parmest.Estimator(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestBlockEF(unittest.TestCase):
    def test_block_ef_structure_counts(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        model = pest._create_scenario_blocks()

        theta_names = model._parmest_theta_names
        self.assertEqual(len(list(model.exp_scenarios.keys())), 2)
        self.assertEqual(
            len(list(model.theta_link_constraints.values())), 2 * len(theta_names)
        )
        self.assertTrue(hasattr(model, "Obj"))
        for block in model.exp_scenarios.values():
            self.assertFalse(block.Total_Cost_Objective.active)

    def test_block_isolation_no_component_leakage(self):
        pest = _build_estimator([(1.0, 2.0), (5.0, 6.0)])
        model = pest._create_scenario_blocks()

        block0 = model.exp_scenarios[0]
        block1 = model.exp_scenarios[1]
        self.assertIsNot(block0.y, block1.y)
        block0.y.set_value(123.0)
        self.assertNotEqual(pyo.value(block1.y), 123.0)
        self.assertNotEqual(pyo.value(block0.x), pyo.value(block1.x))

    def test_fix_theta_sets_all_scenario_theta_values(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        model = pest._create_scenario_blocks(theta_vals={"theta": 1.0}, fix_theta=True)

        self.assertTrue(model.parmest_theta["theta"].fixed)
        self.assertAlmostEqual(pyo.value(model.parmest_theta["theta"]), 1.0, places=10)
        for block in model.exp_scenarios.values():
            self.assertTrue(block.theta.fixed)
            self.assertAlmostEqual(pyo.value(block.theta), 1.0, places=10)

    def test_partial_fix_theta_only_fixed_subset(self):
        class TwoThetaExperiment(Experiment):
            def __init__(self, x, y):
                self.x_data = x
                self.y_data = y
                self.model = None

            def create_model(self):
                m = pyo.ConcreteModel()
                m.theta_a = pyo.Var(initialize=1.0, bounds=(0.0, 4.0))
                m.theta_b = pyo.Var(initialize=0.0, bounds=(-3.0, 3.0))
                m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
                m.y = pyo.Var(initialize=float(self.y_data))
                m.eq = pyo.Constraint(expr=m.y == m.theta_a * m.x + m.theta_b)
                self.model = m

            def label_model(self):
                m = self.model
                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, float(self.y_data))])
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    [
                        (m.theta_a, pyo.ComponentUID(m.theta_a)),
                        (m.theta_b, pyo.ComponentUID(m.theta_b)),
                    ]
                )
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

            def get_labeled_model(self):
                self.create_model()
                self.label_model()
                return self.model

        pest = parmest.Estimator(
            [TwoThetaExperiment(1.0, 3.0), TwoThetaExperiment(2.0, 5.0)],
            obj_function="SSE",
        )
        model = pest._create_scenario_blocks(fixed_theta_values={"theta_a": 2.0})
        self.assertTrue(model.parmest_theta["theta_a"].fixed)
        self.assertFalse(model.parmest_theta["theta_b"].fixed)
        self.assertAlmostEqual(
            pyo.value(model.parmest_theta["theta_a"]), 2.0, places=10
        )
        # only theta_b should be linked across two scenarios
        self.assertEqual(len(list(model.theta_link_constraints.values())), 2)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_objective_at_theta_fixed_value(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        theta_values = pd.DataFrame([[1.0]], columns=["theta"])
        obj_at_theta = pest.objective_at_theta(theta_values=theta_values)
        # residuals at theta=1 are [0, 1], objective is averaged over two scenarios
        self.assertAlmostEqual(obj_at_theta.loc[0, "obj"], 0.5, places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_objective_at_theta_none_uses_initial_theta(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 3.0)])
        obj_at_theta = pest.objective_at_theta()
        # with theta initialized to 0, predictions are [1,2], residuals [1,1], avg objective 1
        self.assertAlmostEqual(obj_at_theta.loc[0, "obj"], 1.0, places=8)
        self.assertAlmostEqual(obj_at_theta.loc[0, "theta"], 0.0, places=8)

    def test_invalid_solver_name_raises_runtimeerror(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        with self.assertRaisesRegex(
            RuntimeError, "Unknown solver in Q_Opt=not_a_solver"
        ):
            pest.theta_est(solver="not_a_solver")

    def test_theta_values_duplicate_columns_rejected(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        duplicate_cols = pd.DataFrame([[1.0, 2.0]], columns=["theta", "theta"])
        with self.assertRaisesRegex(
            ValueError, "Duplicate theta names are not allowed"
        ):
            pest.objective_at_theta(theta_values=duplicate_cols)

    def test_count_total_experiments_multi_output(self):
        exp_list = [
            LinearThetaExperiment(1.0, 2.0, include_second_output=True),
            LinearThetaExperiment(2.0, 4.0, include_second_output=True),
        ]
        total_points = parmest._count_total_experiments(exp_list)
        # The current parmest convention counts datapoints for one output family.
        self.assertEqual(total_points, 2)
