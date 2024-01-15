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


class TestScalarData(unittest.TestCase):
    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0, 1, 2])
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(m.time, m.comp, initialize=1.0)
        return m

    def test_construct_and_get_data(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        data_dict = data.get_data()
        pred_data_dict = {
            pyo.ComponentUID("var[*,A]"): 0.5,
            pyo.ComponentUID("var[*,B]"): 2.0,
        }
        self.assertEqual(data_dict, pred_data_dict)

    def test_construct_exception(self):
        m = self._make_model()
        msg = "Value.*not a scalar"
        with self.assertRaisesRegex(TypeError, msg):
            data = ScalarData({m.var[:, "A"]: [1, 2]})

    def test_eq(self):
        m = self._make_model()
        data1 = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        data2 = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        data3 = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 3.0})
        self.assertEqual(data1, data2)
        self.assertNotEqual(data1, data3)
        data_dict = data2.get_data()
        msg = "not comparable"
        with self.assertRaisesRegex(TypeError, msg):
            data1 == data_dict

    def test_get_data_from_key(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0}, time_set=m.time)
        val = data.get_data_from_key(m.var[:, "A"])
        self.assertEqual(val, 0.5)
        val = data.get_data_from_key(pyo.Reference(m.var[:, "A"]))
        self.assertEqual(val, 0.5)

        val = data.get_data_from_key(m.var[0, "A"])
        self.assertEqual(val, 0.5)

        val = data.get_data_from_key("var[*,A]")
        self.assertEqual(val, 0.5)

    def test_contains_key(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5}, time_set=m.time)
        self.assertTrue(data.contains_key(m.var[:, "A"]))
        self.assertFalse(data.contains_key(m.var[:, "B"]))

    def test_update_data(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        new_data = ScalarData({m.var[:, "A"]: 0.1})
        data.update_data(new_data)
        self.assertEqual(
            data.get_data(),
            {
                pyo.ComponentUID(m.var[:, "A"]): 0.1,
                pyo.ComponentUID(m.var[:, "B"]): 2.0,
            },
        )

        new_data = {m.var[:, "A"]: 0.2}
        data.update_data(new_data)
        self.assertEqual(
            data.get_data(),
            {
                pyo.ComponentUID(m.var[:, "A"]): 0.2,
                pyo.ComponentUID(m.var[:, "B"]): 2.0,
            },
        )

    def test_to_serializable(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        pred_json_dict = {"var[*,A]": 0.5, "var[*,B]": 2.0}
        self.assertEqual(data.to_serializable(), pred_json_dict)

    def test_extract_variables(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        data = data.extract_variables([m.var[:, "A"]])
        pred_data_dict = {pyo.ComponentUID(m.var[:, "A"]): 0.5}
        self.assertEqual(data.get_data(), pred_data_dict)

    def test_extract_variables_exception(self):
        m = self._make_model()
        data = ScalarData({m.var[:, "A"]: 0.5, m.var[:, "B"]: 2.0})
        msg = "extract_variables with copy_values=True"
        with self.assertRaisesRegex(NotImplementedError, msg):
            data = data.extract_variables([m.var[:, "A"]], copy_values=True)


if __name__ == "__main__":
    unittest.main()
