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
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker


class TestVarLinker(unittest.TestCase):
    def _make_models(self, n_time_points_1=3, n_time_points_2=3):
        m1 = pyo.ConcreteModel()
        m1.time = pyo.Set(initialize=range(n_time_points_1))
        m1.comp = pyo.Set(initialize=["A", "B"])
        m1.var = pyo.Var(
            m1.time,
            m1.comp,
            initialize={(i, j): 1.0 + i * 0.1 for i, j in m1.time * m1.comp},
        )
        m1.input = pyo.Var(m1.time, initialize={i: 1.0 - i * 0.1 for i in m1.time})

        m2 = pyo.ConcreteModel()
        m2.time = pyo.Set(initialize=range(n_time_points_2))
        m2.x1 = pyo.Var(m2.time, initialize=2.1)
        m2.x2 = pyo.Var(m2.time, initialize=2.2)
        m2.x3 = pyo.Var(m2.time, initialize=2.3)
        m2.x4 = pyo.Var(m2.time, initialize=2.4)

        return m1, m2

    def test_transfer_one_to_one(self):
        m1, m2 = self._make_models()
        vars1 = [pyo.Reference(m1.var[:, "A"]), pyo.Reference(m1.var[:, "B"]), m1.input]
        vars2 = [m2.x1, m2.x2, m2.x3]

        linker = DynamicVarLinker(vars1, vars2)
        t_source = 0
        t_target = 2
        linker.transfer(t_source=0, t_target=2)

        pred_AB = lambda t: 1.0 + t * 0.1
        pred_input = lambda t: 1.0 - t * 0.1
        for t in m1.time:
            # Both models have same time set

            # Values in source variables have not changed
            self.assertEqual(m1.var[t, "A"].value, pred_AB(t))
            self.assertEqual(m1.var[t, "B"].value, pred_AB(t))
            self.assertEqual(m1.input[t].value, pred_input(t))

            if t == t_target:
                self.assertEqual(m2.x1[t].value, pred_AB(t_source))
                self.assertEqual(m2.x2[t].value, pred_AB(t_source))
                self.assertEqual(m2.x3[t].value, pred_input(t_source))
                self.assertEqual(m2.x4[t].value, 2.4)
            else:
                self.assertEqual(m2.x1[t].value, 2.1)
                self.assertEqual(m2.x2[t].value, 2.2)
                self.assertEqual(m2.x3[t].value, 2.3)
                self.assertEqual(m2.x4[t].value, 2.4)

    def test_transfer_one_to_all(self):
        m1, m2 = self._make_models()
        vars1 = [pyo.Reference(m1.var[:, "A"]), pyo.Reference(m1.var[:, "B"]), m1.input]
        vars2 = [m2.x1, m2.x2, m2.x3]

        linker = DynamicVarLinker(vars1, vars2)
        t_source = 0
        t_target = 2
        linker.transfer(t_source=0, t_target=m2.time)

        pred_AB = lambda t: 1.0 + t * 0.1
        pred_input = lambda t: 1.0 - t * 0.1
        for t in m1.time:
            # Both models have same time set

            # Values in source variables have not changed
            self.assertEqual(m1.var[t, "A"].value, pred_AB(t))
            self.assertEqual(m1.var[t, "B"].value, pred_AB(t))
            self.assertEqual(m1.input[t].value, pred_input(t))

            # Target variables have been updated
            self.assertEqual(m2.x1[t].value, pred_AB(t_source))
            self.assertEqual(m2.x2[t].value, pred_AB(t_source))
            self.assertEqual(m2.x3[t].value, pred_input(t_source))
            self.assertEqual(m2.x4[t].value, 2.4)

    def test_transfer_all_to_all(self):
        m1, m2 = self._make_models()
        vars1 = [pyo.Reference(m1.var[:, "A"]), pyo.Reference(m1.var[:, "B"]), m1.input]
        vars2 = [m2.x1, m2.x2, m2.x3]

        linker = DynamicVarLinker(vars1, vars2)
        t_source = 0
        t_target = 2
        linker.transfer(t_source=m1.time, t_target=m2.time)

        pred_AB = lambda t: 1.0 + t * 0.1
        pred_input = lambda t: 1.0 - t * 0.1
        for t in m1.time:
            # Both models have same time set

            # Values in source variables have not changed
            self.assertEqual(m1.var[t, "A"].value, pred_AB(t))
            self.assertEqual(m1.var[t, "B"].value, pred_AB(t))
            self.assertEqual(m1.input[t].value, pred_input(t))

            # Target variables have been updated
            self.assertEqual(m2.x1[t].value, pred_AB(t))
            self.assertEqual(m2.x2[t].value, pred_AB(t))
            self.assertEqual(m2.x3[t].value, pred_input(t))
            self.assertEqual(m2.x4[t].value, 2.4)

    def test_transfer_exceptions(self):
        m1, m2 = self._make_models()
        vars1 = [pyo.Reference(m1.var[:, "A"]), pyo.Reference(m1.var[:, "B"])]
        vars2 = [m2.x1, m2.x2, m2.x3]

        msg = "must be provided two lists.*of equal length"
        with self.assertRaisesRegex(ValueError, msg):
            linker = DynamicVarLinker(vars1, vars2)

        vars1 = [pyo.Reference(m1.var[:, "A"]), pyo.Reference(m1.var[:, "B"]), m1.input]
        vars2 = [m2.x1, m2.x2, m2.x3]
        linker = DynamicVarLinker(vars1, vars2)
        msg = "Source time points were not provided"
        with self.assertRaisesRegex(RuntimeError, msg):
            linker.transfer(t_target=m2.time)

        msg = "Target time points were not provided"
        with self.assertRaisesRegex(RuntimeError, msg):
            linker.transfer(t_source=m1.time.first())


if __name__ == "__main__":
    unittest.main()
