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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData


class TestDynamicModelInterface(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=range(n_time_points))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time,
            m.comp,
            initialize={(i, j): 1.0 + i * 0.1 for i, j in m.time * m.comp},
        )
        m.input = pyo.Var(m.time, initialize={i: 1.0 - i * 0.1 for i in m.time})
        m.scalar = pyo.Var(initialize=0.5)
        m.var_squared = pyo.Expression(
            m.time,
            m.comp,
            initialize={(i, j): m.var[i, j] ** 2 for i, j in m.time * m.comp},
        )
        return m

    def _hashRef(self, reference):
        return tuple(id(obj) for obj in reference.values())

    def test_interface_construct(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        scalar_vars = interface.get_scalar_variables()
        self.assertEqual(len(scalar_vars), 1)
        self.assertIs(scalar_vars[0], m.scalar)

        dae_vars = interface.get_indexed_variables()
        self.assertEqual(len(dae_vars), 3)
        dae_var_set = set(self._hashRef(var) for var in dae_vars)
        pred_dae_var = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
            m.input,
        ]
        for var in pred_dae_var:
            self.assertIn(self._hashRef(var), dae_var_set)

        dae_expr = interface.get_indexed_expressions()
        dae_expr_set = set(self._hashRef(expr) for expr in dae_expr)
        self.assertEqual(len(dae_expr), 2)
        pred_dae_expr = [
            pyo.Reference(m.var_squared[:, "A"]),
            pyo.Reference(m.var_squared[:, "B"]),
        ]
        for expr in pred_dae_expr:
            self.assertIn(self._hashRef(expr), dae_expr_set)

    def test_get_scalar_var_data(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_scalar_variable_data()
        self.assertEqual({pyo.ComponentUID(m.scalar): 0.5}, data)

    def test_get_data_at_time_all_points(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(include_expr=True)
        pred_data = TimeSeriesData(
            {
                m.var[:, "A"]: [1.0, 1.1, 1.2],
                m.var[:, "B"]: [1.0, 1.1, 1.2],
                m.input[:]: [1.0, 0.9, 0.8],
                m.var_squared[:, "A"]: [1.0**2, 1.1**2, 1.2**2],
                m.var_squared[:, "B"]: [1.0**2, 1.1**2, 1.2**2],
            },
            m.time,
        )
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_subset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(time=[0, 2])
        pred_data = TimeSeriesData(
            {
                m.var[:, "A"]: [1.0, 1.2],
                m.var[:, "B"]: [1.0, 1.2],
                m.input[:]: [1.0, 0.8],
            },
            [0, 2],
        )
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_singleton(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(time=1, include_expr=True)
        pred_data = ScalarData(
            {
                m.var[:, "A"]: 1.1,
                m.var[:, "B"]: 1.1,
                m.input[:]: 0.9,
                m.var_squared[:, "A"]: 1.1**2,
                m.var_squared[:, "B"]: 1.1**2,
            }
        )
        self.assertEqual(data, pred_data)

    def test_load_scalar_data(self):
        # load_scalar_data has been removed. Instead we simply call
        # load_data
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.scalar): 6.0}
        interface.load_data(data)
        self.assertEqual(m.scalar.value, 6.0)

    def test_load_data_at_time_all(self):
        # NOTE: load_data_at_time has been deprecated
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 5.5)
            self.assertEqual(m.input[t].value, 6.6)

    def test_load_data_at_time_subset(self):
        # NOTE: load_data_at_time has been deprecated
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        old_A = {t: m.var[t, "A"].value for t in m.time}
        old_input = {t: m.input[t].value for t in m.time}

        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        time_points = [1, 2]
        time_set = set(time_points)
        interface.load_data(data, time_points=[1, 2])

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            if t in time_set:
                self.assertEqual(m.var[t, "A"].value, 5.5)
                self.assertEqual(m.input[t].value, 6.6)
            else:
                self.assertEqual(m.var[t, "A"].value, old_A[t])
                self.assertEqual(m.input[t].value, old_input[t])

    def test_load_data_from_dict_scalar_var(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.scalar): 6.0}
        interface.load_data(data)
        self.assertEqual(m.scalar.value, 6.0)

    def test_load_data_from_dict_indexed_var(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.input): 6.0}
        interface.load_data(data)
        for t in m.time:
            self.assertEqual(m.input[t].value, 6.0)

    def test_load_data_from_dict_indexed_var_list_data(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data_list = [2, 3, 4]
        data = {pyo.ComponentUID(m.input): data_list}
        # Need to provide data to load_data that can be interpreted
        # as a TimeSeriesData
        interface.load_data((data, m.time))
        for i, t in enumerate(m.time):
            self.assertEqual(m.input[t].value, data_list[i])

    def test_load_data_from_ScalarData_to_point(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        interface.load_data(data, time_points=1)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        old_A = [1.0, 1.1, 1.2]
        old_input = [1.0, 0.9, 0.8]
        for i, t in enumerate(m.time):
            if t == 1:
                self.assertEqual(m.var[t, "A"].value, 5.5)
                self.assertEqual(m.input[t].value, 6.6)
            else:
                self.assertEqual(m.var[t, "A"].value, old_A[i])
                self.assertEqual(m.input[t].value, old_input[i])

    def test_load_data_from_ScalarData_toall(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 5.5)
            self.assertEqual(m.input[t].value, 6.6)

    def test_load_data_from_ScalarData_tosubset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        old_A = {t: m.var[t, "A"].value for t in m.time}
        old_input = {t: m.input[t].value for t in m.time}

        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        time_points = [1, 2]
        time_set = set(time_points)
        interface.load_data(data, time_points=[1, 2])

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            if t in time_set:
                self.assertEqual(m.var[t, "A"].value, 5.5)
                self.assertEqual(m.input[t].value, 6.6)
            else:
                self.assertEqual(m.var[t, "A"].value, old_A[t])
                self.assertEqual(m.input[t].value, old_input[t])

    def test_load_data_from_TimeSeriesData(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        new_A = [1.0, 2.0, 3.0]
        new_input = [4.0, 5.0, 6.0]
        data = TimeSeriesData({m.var[:, "A"]: new_A, m.input[:]: new_input}, m.time)
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for i, t in enumerate(m.time):
            self.assertEqual(m.var[t, "A"].value, new_A[i])
            self.assertEqual(m.input[t].value, new_input[i])

    def test_load_data_from_TimeSeriesData_tuple(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        new_A = [1.0, 2.0, 3.0]
        new_input = [4.0, 5.0, 6.0]
        data = ({m.var[:, "A"]: new_A, m.input[:]: new_input}, m.time)
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for i, t in enumerate(m.time):
            self.assertEqual(m.var[t, "A"].value, new_A[i])
            self.assertEqual(m.input[t].value, new_input[i])

    def test_load_data_from_IntervalData(self):
        m = self._make_model(5)
        interface = DynamicModelInterface(m, m.time)
        new_A = [-1.1, -1.2, -1.3]
        new_input = [3.0, 2.9, 2.8]
        data = IntervalData(
            {m.var[:, "A"]: new_A, m.input[:]: new_input},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        interface.load_data(data)
        B_data = [m.var[t, "B"].value for t in m.time]
        self.assertEqual(B_data, [1.0, 1.1, 1.2, 1.3, 1.4])
        for t in m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            self.assertEqual(m.var[t, "A"].value, new_A[idx])
            self.assertEqual(m.input[t].value, new_input[idx])

    def test_load_data_from_IntervalData_tuple(self):
        m = self._make_model(5)
        interface = DynamicModelInterface(m, m.time)
        new_A = [-1.1, -1.2, -1.3]
        new_input = [3.0, 2.9, 2.8]
        data = (
            {m.var[:, "A"]: new_A, m.input[:]: new_input},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        interface.load_data(data)
        B_data = [m.var[t, "B"].value for t in m.time]
        self.assertEqual(B_data, [1.0, 1.1, 1.2, 1.3, 1.4])
        for t in m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            self.assertEqual(m.var[t, "A"].value, new_A[idx])
            self.assertEqual(m.input[t].value, new_input[idx])

    def test_load_data_bad_arg(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        new_A = [1.0, 2.0, 3.0]
        new_input = [4.0, 5.0, 6.0]
        data = ({m.var[:, "A"]: new_A, m.input[:]: new_input}, m.time)
        msg = "can only be set if data is IntervalData-compatible"
        with self.assertRaisesRegex(RuntimeError, msg):
            interface.load_data(data, prefer_left=True)

    def test_copy_values_at_time_default(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        interface.copy_values_at_time()
        # Default is to copy values from t0 to all points in time
        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 1.0)
            self.assertEqual(m.var[t, "B"].value, 1.0)
            self.assertEqual(m.input[t].value, 1.0)

    def test_copy_values_at_time_toall(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        interface.copy_values_at_time(source_time=tf)
        # Default is to copy values to all points in time
        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 1.2)
            self.assertEqual(m.var[t, "B"].value, 1.2)
            self.assertEqual(m.input[t].value, 0.8)

    def test_copy_values_at_time_tosubset(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        target_points = [t for t in m.time if t != m.time.first()]
        target_subset = set(target_points)
        interface.copy_values_at_time(source_time=tf, target_time=target_points)
        # Default is to copy values to all points in time
        for t in m.time:
            if t in target_subset:
                self.assertEqual(m.var[t, "A"].value, 1.2)
                self.assertEqual(m.var[t, "B"].value, 1.2)
                self.assertEqual(m.input[t].value, 0.8)
            else:
                # t0 has not been altered.
                self.assertEqual(m.var[t, "A"].value, 1.0)
                self.assertEqual(m.var[t, "B"].value, 1.0)
                self.assertEqual(m.input[t].value, 1.0)

    def test_copy_values_at_time_exception(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        msg = "copy_values_at_time can only copy"
        with self.assertRaisesRegex(ValueError, msg):
            interface.copy_values_at_time(source_time=m.time, target_time=tf)

    def test_shift_values_by_time(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        dt = 1.0
        interface.shift_values_by_time(dt)

        t = 0
        self.assertEqual(m.var[t, "A"].value, 1.1)
        self.assertEqual(m.var[t, "B"].value, 1.1)
        self.assertEqual(m.input[t].value, 0.9)

        t = 1
        self.assertEqual(m.var[t, "A"].value, 1.2)
        self.assertEqual(m.var[t, "B"].value, 1.2)
        self.assertEqual(m.input[t].value, 0.8)

        t = 2
        # For values within dt of the endpoint, the value at
        # the boundary is copied.
        self.assertEqual(m.var[t, "A"].value, 1.2)
        self.assertEqual(m.var[t, "B"].value, 1.2)
        self.assertEqual(m.input[t].value, 0.8)

    def test_get_penalty_from_constant_target(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        setpoint_data = ScalarData({m.var[:, "A"]: 1.0, m.var[:, "B"]: 2.0})
        weight_data = ScalarData({m.var[:, "A"]: 10.0, m.var[:, "B"]: 0.1})

        vset, tr_cost = interface.get_penalty_from_target(
            setpoint_data, weight_data=weight_data
        )
        m.var_set = vset
        m.tracking_cost = tr_cost
        for t in m.time:
            for i in m.var_set:
                pred_expr = (
                    10.0 * (m.var[t, "A"] - 1.0) ** 2
                    if i == 0
                    else 0.1 * (m.var[t, "B"] - 2.0) ** 2
                )
                self.assertEqual(pyo.value(pred_expr), pyo.value(m.tracking_cost[i, t]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[i, t].expr)
                )

    def test_get_penalty_from_constant_target_var_subset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        setpoint_data = ScalarData(
            {m.var[:, "A"]: 1.0, m.var[:, "B"]: 2.0, m.input[:]: 3.0}
        )
        weight_data = ScalarData(
            {m.var[:, "A"]: 10.0, m.var[:, "B"]: 0.1, m.input[:]: 0.01}
        )

        variables = [m.var[:, "A"], m.var[:, "B"]]
        m.variable_set = pyo.Set(initialize=range(len(variables)))
        new_set, tr_cost = interface.get_penalty_from_target(
            setpoint_data,
            variables=variables,
            weight_data=weight_data,
            variable_set=m.variable_set,
        )
        m.tracking_cost = tr_cost
        self.assertIs(m.variable_set, new_set)
        for t in m.time:
            for i in m.variable_set:
                pred_expr = (
                    10.0 * (m.var[t, "A"] - 1.0) ** 2
                    if i == 0
                    else +0.1 * (m.var[t, "B"] - 2.0) ** 2
                )
                self.assertEqual(pyo.value(pred_expr), pyo.value(m.tracking_cost[i, t]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[i, t].expr)
                )


class TestGetPenaltyFromTarget(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        return m

    def test_constant_setpoint(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        setpoint = {m.var[:, "A"]: 0.3, m.var[:, "B"]: 0.4}
        m.var_set, m.penalty = interface.get_penalty_from_target(setpoint)

        # Note that the order of the variables here is not deterministic
        # in some Python <=3.6 implementations
        pred_expr = {
            (i, t): (
                (m.var[t, "A"] - 0.3) ** 2 if i == 0 else (m.var[t, "B"] - 0.4) ** 2
            )
            for i, t in m.var_set * m.time
        }
        for t in m.time:
            for i in m.var_set:
                self.assertTrue(
                    compare_expressions(pred_expr[i, t], m.penalty[i, t].expr)
                )
                self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))

    def test_varying_setpoint(self):
        m = self._make_model(n_time_points=5)
        interface = DynamicModelInterface(m, m.time)
        A_target = [0.4, 0.6, 0.1, 0.0, 1.1]
        B_target = [0.8, 0.9, 1.3, 1.5, 1.4]
        setpoint = ({m.var[:, "A"]: A_target, m.var[:, "B"]: B_target}, m.time)
        m.var_set, m.penalty = interface.get_penalty_from_target(setpoint)

        target = {
            (i, t): A_target[j] if i == 0 else B_target[t]
            for i in m.var_set
            for (j, t) in enumerate(m.time)
        }
        for i, t in m.var_set * m.time:
            var = m.var[t, "A"] if i == 0 else m.var[t, "B"]
            pred_expr = (var - target[i, t]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_piecewise_constant_setpoint(self):
        m = self._make_model(n_time_points=5)
        interface = DynamicModelInterface(m, m.time)
        A_target = [0.3, 0.9, 0.7]
        B_target = [1.1, 0.1, 0.5]
        setpoint = (
            {m.var[:, "A"]: A_target, m.var[:, "B"]: B_target},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        m.var_set, m.penalty = interface.get_penalty_from_target(setpoint)
        target = {
            (i, j): A_target[j] if i == 0 else B_target[j]
            for i in m.var_set
            for j in range(len(A_target))
        }
        for i, t in m.var_set * m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            var = m.var[t, "A"] if i == 0 else m.var[t, "B"]
            pred_expr = (var - target[i, idx]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_piecewise_constant_setpoint_with_specified_variables(self):
        m = self._make_model(n_time_points=5)
        interface = DynamicModelInterface(m, m.time)
        A_target = [0.3, 0.9, 0.7]
        B_target = [1.1, 0.1, 0.5]
        setpoint = (
            {m.var[:, "A"]: A_target, m.var[:, "B"]: B_target},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        variables = [pyo.Reference(m.var[:, "B"])]
        m.var_set, m.penalty = interface.get_penalty_from_target(
            setpoint, variables=variables
        )
        self.assertEqual(len(m.var_set), 1)
        self.assertEqual(m.var_set[1], 0)
        for i, t in m.var_set * m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            var = m.var[t, "B"]
            pred_expr = (var - B_target[idx]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_piecewise_constant_setpoint_time_subset(self):
        m = self._make_model(n_time_points=5)
        interface = DynamicModelInterface(m, m.time)
        A_target = [0.3, 0.9, 0.7]
        B_target = [1.1, 0.1, 0.5]
        setpoint = (
            {m.var[:, "A"]: A_target, m.var[:, "B"]: B_target},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        m.sample_points = pyo.Set(initialize=[0.0, 2.0, 4.0])
        m.var_set, m.penalty = interface.get_penalty_from_target(
            setpoint, time=m.sample_points
        )
        idx_sets = ComponentSet(m.penalty.index_set().subsets())
        self.assertIn(m.var_set, idx_sets)
        self.assertIn(m.sample_points, idx_sets)
        target = {
            (i, j): A_target[j] if i == 0 else B_target[j]
            for i in m.var_set
            for j in range(len(A_target))
        }
        for i, t in m.var_set * m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            if t in m.sample_points:
                var = m.var[t, "A"] if i == 0 else m.var[t, "B"]
                pred_expr = (var - target[i, idx]) ** 2
                self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
                self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))
            else:
                self.assertNotIn((i, t), m.penalty)


if __name__ == "__main__":
    unittest.main()
