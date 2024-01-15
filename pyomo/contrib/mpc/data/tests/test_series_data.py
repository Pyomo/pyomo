#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData


class TestSeriesData(unittest.TestCase):
    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.0, 0.1, 0.2])
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(m.time, m.comp, initialize=1.0)
        return m

    def test_construct_and_get_data(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)

        processed_data_dict = {
            pyo.ComponentUID(key): val for key, val in data_dict.items()
        }
        self.assertEqual(data.get_data(), processed_data_dict)

    def test_construct_exception(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4]}
        msg = "must have same length"
        with self.assertRaisesRegex(ValueError, msg):
            data = TimeSeriesData(data_dict, m.time)

        data_dict = {m.var[:, "A"]: [1, 2], m.var[:, "B"]: [2, 4]}
        with self.assertRaisesRegex(ValueError, msg):
            # series don't have same lengths as time
            data = TimeSeriesData(data_dict, m.time)

        msg = "not sorted"
        with self.assertRaisesRegex(ValueError, msg):
            # Time list has right number of points, but is not sorted
            # increasing.
            data = TimeSeriesData(data_dict, [0, -1])

    def test_get_time_points(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)
        self.assertEqual(data.get_time_points(), list(m.time))

        new_time_list = [3, 4, 5]
        data = TimeSeriesData(data_dict, new_time_list)
        self.assertEqual(data.get_time_points(), new_time_list)

    def test_get_data_at_time_indices(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)
        new_data = data.get_data_at_time_indices(1)
        self.assertEqual(ScalarData({m.var[:, "A"]: 2, m.var[:, "B"]: 4}), new_data)

        new_data = data.get_data_at_time_indices([1])
        t1 = m.time.at(2)  # Sets are indexed starting from 1...
        self.assertEqual(
            TimeSeriesData({m.var[:, "A"]: [2], m.var[:, "B"]: [4]}, [t1]), new_data
        )

        new_t = [m.time.at(1), m.time.at(3)]
        new_data = data.get_data_at_time_indices([0, 2])
        self.assertEqual(
            TimeSeriesData({m.var[:, "A"]: [1, 3], m.var[:, "B"]: [2, 6]}, new_t),
            new_data,
        )

    def test_get_data_at_time(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)
        new_data = data.get_data_at_time(0.1)
        self.assertEqual(ScalarData({m.var[:, "A"]: 2, m.var[:, "B"]: 4}), new_data)

        t1 = 0.1
        new_data = data.get_data_at_time([t1])
        self.assertEqual(
            TimeSeriesData({m.var[:, "A"]: [2], m.var[:, "B"]: [4]}, [t1]), new_data
        )

        new_t = [0.0, 0.2]
        new_data = data.get_data_at_time(new_t)
        self.assertEqual(
            TimeSeriesData({m.var[:, "A"]: [1, 3], m.var[:, "B"]: [2, 6]}, new_t),
            new_data,
        )

    def test_get_data_at_time_with_tolerance(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)

        # Test an invalid time value. A tolerance of None gives us
        # the closest index
        new_data = data.get_data_at_time(-0.1, tolerance=None)
        self.assertEqual(ScalarData({m.var[:, "A"]: 1, m.var[:, "B"]: 2}), new_data)

        # Test a value that is only valid within tolerance
        new_data = data.get_data_at_time(-0.0001, tolerance=1e-3)
        self.assertEqual(ScalarData({m.var[:, "A"]: 1, m.var[:, "B"]: 2}), new_data)

        # The default is to raise an error in the case of any discrepancy.
        msg = "Time point.*is invalid"
        with self.assertRaisesRegex(RuntimeError, msg):
            new_data = data.get_data_at_time(-0.0001)

        # Test a value that is invalid within tolerance
        msg = "Time point.*is invalid"
        with self.assertRaisesRegex(RuntimeError, msg):
            new_data = data.get_data_at_time(-0.01, tolerance=1e-3)

    def test_to_serializable(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time).to_serializable()

        pred_json_obj = (
            {str(pyo.ComponentUID(key)): val for key, val in data_dict.items()},
            list(m.time),
        )
        self.assertEqual(data, pred_json_obj)

        # Test attributes of the TimeSeriesTuple namedtuple
        self.assertEqual(data.time, list(m.time))
        self.assertEqual(
            data.data,
            {str(pyo.ComponentUID(key)): val for key, val in data_dict.items()},
        )

    def test_concatenate(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        time1 = [t for t in m.time]
        data1 = TimeSeriesData(data_dict, time1)

        time2 = [t + 1.0 for t in m.time]
        data2 = TimeSeriesData(data_dict, time2)

        data1.concatenate(data2)
        pred_time = time1 + time2
        pred_data = {
            m.var[:, "A"]: [1, 2, 3, 1, 2, 3],
            m.var[:, "B"]: [2, 4, 6, 2, 4, 6],
        }
        # Note that data1 has been modified in place
        self.assertEqual(TimeSeriesData(pred_data, pred_time), data1)

    def test_concatenate_exception(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        time1 = [t for t in m.time]
        data1 = TimeSeriesData(data_dict, time1)

        msg = "Initial time point.*is not greater"
        with self.assertRaisesRegex(ValueError, msg):
            data1.concatenate(data1)

    def test_shift_time_points(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)

        offset = 1.0
        data.shift_time_points(offset)
        self.assertEqual(data.get_time_points(), [t + offset for t in m.time])

    def test_extract_variables(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)

        new_data = data.extract_variables([m.var[:, "A"]])
        self.assertEqual(new_data, TimeSeriesData({m.var[:, "A"]: [1, 2, 3]}, m.time))

    def test_shift_then_get_data(self):
        m = self._make_model()
        data_dict = {m.var[:, "A"]: [1, 2, 3], m.var[:, "B"]: [2, 4, 6]}
        data = TimeSeriesData(data_dict, m.time)

        offset = 0.1
        data.shift_time_points(offset)
        self.assertEqual(data.get_time_points(), [t + offset for t in m.time])

        # A time point of zero is no longer valid
        msg = "Time point.*is invalid"
        with self.assertRaisesRegex(RuntimeError, msg):
            t0_data = data.get_data_at_time(0.0, tolerance=1e-3)

        t1_data = data.get_data_at_time(0.1)
        self.assertEqual(t1_data, ScalarData({m.var[:, "A"]: 1, m.var[:, "B"]: 2}))


if __name__ == "__main__":
    unittest.main()
