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
from pyomo.contrib.mpc.data.series_data import TimeSeriesData


class TestSeriesData(unittest.TestCase):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0, 1, 2])
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


if __name__ == "__main__":
    unittest.main()
