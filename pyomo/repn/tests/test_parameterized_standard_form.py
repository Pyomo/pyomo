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
#

from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
import pyomo.common.unittest as unittest

from pyomo.environ import ConcreteModel, Constraint, Var
from pyomo.core.expr import (
    MonomialTermExpression,
    NegationExpression,
    ProductExpression,
)
from pyomo.core.expr.compare import assertExpressionsEqual

from pyomo.repn.plugins.parameterized_standard_form import (
    ParameterizedLinearStandardFormCompiler,
    _CSRMatrix,
    _CSCMatrix,
)


@unittest.skipUnless(
    numpy_available & scipy_available,
    "CSC and CSR representations require scipy and numpy",
)
class TestSparseMatrixRepresentations(unittest.TestCase):
    def test_csr_to_csc_only_data(self):
        A = _CSRMatrix([5, 8, 3, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4], 4, 4)
        thing = A.tocsc()

        self.assertTrue(np.all(thing.data == np.array([5, 8, 6, 3])))
        self.assertTrue(np.all(thing.indices == np.array([0, 1, 3, 2])))
        self.assertTrue(np.all(thing.indptr == np.array([0, 1, 3, 4, 4])))

    def test_csr_to_csc_pyomo_exprs(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        A = _CSRMatrix(
            [5, 8 * m.x, 3 * m.x * m.y**2, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4], 4, 4
        )
        thing = A.tocsc()

        self.assertEqual(thing.data[0], 5)
        assertExpressionsEqual(self, thing.data[1], 8 * m.x)
        self.assertEqual(thing.data[2], 6)
        assertExpressionsEqual(self, thing.data[3], 3 * m.x * m.y**2)
        self.assertEqual(thing.data.shape, (4,))

        self.assertTrue(np.all(thing.indices == np.array([0, 1, 3, 2])))
        self.assertTrue(np.all(thing.indptr == np.array([0, 1, 3, 4, 4])))

    def test_csr_to_csc_empty_matrix(self):
        A = _CSRMatrix([], [], [0], 0, 4)
        thing = A.tocsc()

        self.assertEqual(thing.data.size, 0)
        self.assertEqual(thing.indices.size, 0)
        self.assertEqual(thing.shape, (0, 4))
        self.assertTrue(np.all(thing.indptr == np.zeros(5)))


def assertExpressionArraysEqual(self, A, B):
    self.assertEqual(A.shape, B.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assertExpressionsEqual(self, A[i, j], B[i, j])


def assertExpressionListsEqual(self, A, B):
    self.assertEqual(len(A), len(B))
    for i, a in enumerate(A):
        assertExpressionsEqual(self, a, B[i])


@unittest.skipUnless(
    numpy_available & scipy_available,
    "Parameterized standard form requires scipy and numpy",
)
class TestParameterizedStandardFormCompiler(unittest.TestCase):
    def test_linear_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.c = Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        repn = ParameterizedLinearStandardFormCompiler().write(m)

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[-1, -2, 0], [0, 1, 4]])))
        self.assertTrue(np.all(repn.rhs == np.array([-3, 5])))
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1], m.y[3]])

    def test_parameterized_linear_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.data = Var([1, 2])
        m.more_data = Var()
        m.c = Constraint(expr=m.x + 2 * m.data[1] * m.data[2] * m.y[1] >= 3)
        m.d = Constraint(expr=m.y[1] + 4 * m.y[3] <= 5 * m.more_data)

        repn = ParameterizedLinearStandardFormCompiler().write(
            m, wrt=[m.data, m.more_data]
        )

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        assertExpressionArraysEqual(
            self,
            repn.A.todense(),
            np.array(
                [
                    [
                        -1,
                        NegationExpression(
                            (
                                ProductExpression(
                                    [MonomialTermExpression([2, m.data[1]]), m.data[2]]
                                ),
                            )
                        ),
                        0,
                    ],
                    [0, 1, 4],
                ]
            ),
        )
        assertExpressionListsEqual(self, repn.rhs, [-3, 5 * m.more_data])
        self.assertEqual(repn.rows, [(m.c, -1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1], m.y[3]])

    def test_parameterized_almost_dense_linear_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.data = Var([1, 2])
        m.more_data = Var()
        m.c = Constraint(
            expr=m.x + 2 * m.y[1] + 4 * m.y[3] + m.more_data >= 10 * m.data[1] ** 2
        )
        m.d = Constraint(expr=5 * m.x + 6 * m.y[1] + 8 * m.data[2] * m.y[3] <= 20)

        repn = ParameterizedLinearStandardFormCompiler().write(
            m, wrt=[m.data, m.more_data]
        )

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        # m.c gets interpretted as a <= Constraint, and you can't really blame
        # pyomo for that because it's not parameterized yet. So that's why this
        # differs from the test in test_standard_form.py
        assertExpressionArraysEqual(
            self, repn.A.todense(), np.array([[-1, -2, -4], [5, 6, 8 * m.data[2]]])
        )
        assertExpressionListsEqual(
            self, repn.rhs, [-(10 * m.data[1] ** 2 - m.more_data), 20]
        )
        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1)])
        self.assertEqual(repn.columns, [m.x, m.y[1], m.y[3]])

    def test_parameterized_linear_model_row_col_order(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        m.data = Var([1, 2])
        m.more_data = Var()
        m.c = Constraint(expr=m.x + 2 * m.data[1] * m.data[2] * m.y[1] >= 3)
        m.d = Constraint(expr=m.y[1] + 4 * m.y[3] <= 5 * m.more_data)

        repn = ParameterizedLinearStandardFormCompiler().write(
            m,
            wrt=[m.data, m.more_data],
            column_order=[m.y[3], m.y[2], m.x, m.y[1]],
            row_order=[m.d, m.c],
        )

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        assertExpressionArraysEqual(
            self,
            repn.A.todense(),
            np.array(
                [
                    [4, 0, 1],
                    [
                        0,
                        -1,
                        NegationExpression(
                            (
                                ProductExpression(
                                    [MonomialTermExpression([2, m.data[1]]), m.data[2]]
                                ),
                            )
                        ),
                    ],
                ]
            ),
        )
        assertExpressionListsEqual(self, repn.rhs, np.array([5 * m.more_data, -3]))
        self.assertEqual(repn.rows, [(m.d, 1), (m.c, -1)])
        self.assertEqual(repn.columns, [m.y[3], m.x, m.y[1]])
