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
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid


class TestGetCUID(unittest.TestCase):
    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.0, 0.1, 0.2])
        m.space = pyo.Set(initialize=[1.0, 1.5, 2.0])
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(m.time, m.comp, initialize=1.0)
        m.txc_var = pyo.Var(m.time, m.space, m.comp, initialize=2.0)

        @m.Block(m.time, m.space)
        def b(b, t, x):
            b.bvar1 = pyo.Var(initialize=3.0)
            b.bvar2 = pyo.Var(m.comp, initialize=3.0)

        return m

    def test_get_cuid(self):
        m = self._make_model()

        pred_cuid = pyo.ComponentUID(m.var[:, "A"])
        self.assertEqual(get_indexed_cuid(m.var[:, "A"]), pred_cuid)
        self.assertEqual(get_indexed_cuid(pyo.Reference(m.var[:, "A"])), pred_cuid)
        self.assertEqual(get_indexed_cuid("var[*,A]"), pred_cuid)
        self.assertEqual(get_indexed_cuid("var[*,'A']"), pred_cuid)
        self.assertEqual(get_indexed_cuid(m.var[0, "A"], sets=(m.time,)), pred_cuid)

    def test_get_cuid_twosets(self):
        m = self._make_model()

        pred_cuid = pyo.ComponentUID(m.b[:, :].bvar2["A"])
        self.assertEqual(get_indexed_cuid(m.b[:, :].bvar2["A"]), pred_cuid)
        self.assertEqual(
            get_indexed_cuid(pyo.Reference(m.b[:, :].bvar2["A"])), pred_cuid
        )
        self.assertEqual(get_indexed_cuid("b[*,*].bvar2[A]"), pred_cuid)
        self.assertEqual(
            get_indexed_cuid(m.b[0, 1].bvar2["A"], sets=(m.time, m.space)), pred_cuid
        )

    def test_get_cuid_dereference(self):
        m = self._make_model()
        m.ref = pyo.Reference(m.var[:, "A"])
        m.ref2 = pyo.Reference(m.ref)

        pred_cuid = pyo.ComponentUID(m.var[:, "A"])

        # ref is attached to the model, so by default we don't "dereference"
        self.assertNotEqual(get_indexed_cuid(m.ref), pred_cuid)
        self.assertEqual(get_indexed_cuid(m.ref), pyo.ComponentUID(m.ref[:]))

        # If we use dereference=True, we do dereference and get the CUID of
        # the underlying slice.
        self.assertEqual(get_indexed_cuid(m.ref, dereference=True), pred_cuid)

        # However, we only dereference once, so a reference-to-reference
        # does not reveal the underlying slice (of the original reference)
        self.assertNotEqual(get_indexed_cuid(m.ref2, dereference=True), pred_cuid)
        self.assertEqual(
            get_indexed_cuid(m.ref2, dereference=True), pyo.ComponentUID(m.ref[:])
        )

        # But if we use dereference=2, we allow two dereferences, and get
        # the original slice
        self.assertEqual(get_indexed_cuid(m.ref2, dereference=2), pred_cuid)

    def test_get_cuid_context(self):
        m = self._make_model()
        top = pyo.ConcreteModel()
        top.m = m

        pred_cuid = pyo.ComponentUID(m.var[:, "A"], context=m)
        self.assertEqual(get_indexed_cuid(m.var[:, "A"], context=m), pred_cuid)
        self.assertEqual(
            get_indexed_cuid(pyo.Reference(m.var[:, "A"]), context=m), pred_cuid
        )

        # This is what we would expect without a context arg
        full_cuid = pyo.ComponentUID(m.var[:, "A"])
        self.assertNotEqual(get_indexed_cuid("m.var[*,A]"), pred_cuid)
        self.assertEqual(get_indexed_cuid("m.var[*,A]"), full_cuid)

        msg = "Context is not allowed"
        with self.assertRaisesRegex(ValueError, msg):
            # Passing context with a string raises a reasonable
            # error from the CUID constructor
            get_indexed_cuid("m.var[*,A]", context=m)

        self.assertEqual(
            get_indexed_cuid(m.var[0, "A"], sets=(m.time,), context=m), pred_cuid
        )


if __name__ == "__main__":
    unittest.main()
