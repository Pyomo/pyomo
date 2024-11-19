#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.environ import ConcreteModel, Var, Constraint


@unittest.skipUnless(numpy_available, "tests require numpy")
class TestNumpyExpr(unittest.TestCase):
    def test_scalar_operations(self):
        m = ConcreteModel()
        m.x = Var()

        a = np.array(m.x)
        self.assertEqual(a.shape, ())

        self.assertExpressionsEqual(5 * a, 5 * m.x)
        self.assertExpressionsEqual(np.array([2, 3]) * a, [2 * m.x, 3 * m.x])
        self.assertExpressionsEqual(np.array([5, 6]) * m.x, [5 * m.x, 6 * m.x])
        self.assertExpressionsEqual(np.array([8, m.x]) * m.x, [8 * m.x, m.x * m.x])

        a = np.array([m.x])
        self.assertEqual(a.shape, (1,))

        self.assertExpressionsEqual(5 * a, [5 * m.x])
        self.assertExpressionsEqual(np.array([2, 3]) * a, [2 * m.x, 3 * m.x])
        self.assertExpressionsEqual(np.array([5, 6]) * m.x, [5 * m.x, 6 * m.x])
        self.assertExpressionsEqual(np.array([8, m.x]) * m.x, [8 * m.x, m.x * m.x])

    def test_vector_operations(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([0, 1, 2])

        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            # TODO: when we finally support a true matrix expression
            # system, this test should work
            self.assertExpressionsEqual(5 * m.y, [5 * m.y[0], 5 * m.y[1], 5 * m.y[2]])

        a = np.array(5)
        self.assertExpressionsEqual(a * m.y, [5 * m.y[0], 5 * m.y[1], 5 * m.y[2]])
        self.assertExpressionsEqual(m.y * a, [5 * m.y[0], 5 * m.y[1], 5 * m.y[2]])
        a = np.array([5])
        self.assertExpressionsEqual(a * m.y, [5 * m.y[0], 5 * m.y[1], 5 * m.y[2]])
        self.assertExpressionsEqual(m.y * a, [5 * m.y[0], 5 * m.y[1], 5 * m.y[2]])

        a = np.array(5)
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            # TODO: when we finally support a true matrix expression
            # system, this test should work
            self.assertExpressionsEqual(
                a * m.x * m.y, [5 * m.x * m.y[0], 5 * m.x * m.y[1], 5 * m.x * m.y[2]]
            )
        self.assertExpressionsEqual(
            a * m.y * m.x, [5 * m.y[0] * m.x, 5 * m.y[1] * m.x, 5 * m.y[2] * m.x]
        )
        self.assertExpressionsEqual(
            a * m.y * m.y,
            [5 * m.y[0] * m.y[0], 5 * m.y[1] * m.y[1], 5 * m.y[2] * m.y[2]],
        )
        self.assertExpressionsEqual(
            m.y * a * m.x, [5 * m.y[0] * m.x, 5 * m.y[1] * m.x, 5 * m.y[2] * m.x]
        )
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            # TODO: when we finally support a true matrix expression
            # system, this test should work
            self.assertExpressionsEqual(
                m.y * m.x * a, [5 * m.y[0] * m.x, 5 * m.y[1] * m.x, 5 * m.y[2] * m.x]
            )
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            # TODO: when we finally support a true matrix expression
            # system, this test should work
            self.assertExpressionsEqual(
                m.x * a * m.y, [5 * m.y[0] * m.x, 5 * m.y[1] * m.x, 5 * m.y[2] * m.x]
            )
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            # TODO: when we finally support a true matrix expression
            # system, this test should work
            self.assertExpressionsEqual(
                m.x * m.y * a, [5 * m.y[0] * m.x, 5 * m.y[1] * m.x, 5 * m.y[2] * m.x]
            )


if __name__ == "__main__":
    unittest.main()
