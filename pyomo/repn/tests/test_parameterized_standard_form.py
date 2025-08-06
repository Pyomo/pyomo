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
#

from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    inequality,
    Objective,
    maximize,
    Var,
)
from pyomo.core.expr import (
    MonomialTermExpression,
    NegationExpression,
    ProductExpression,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)

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
        A = _CSRMatrix(([5, 8, 3, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4]), [4, 4])
        thing = A.tocsc()

        self.assertTrue(np.all(thing.data == np.array([5, 8, 6, 3])))
        self.assertTrue(np.all(thing.indices == np.array([0, 1, 3, 2])))
        self.assertTrue(np.all(thing.indptr == np.array([0, 1, 3, 4, 4])))

        should_be_A = thing.tocsr()
        self.assertTrue(np.all(should_be_A.data == A.data))
        self.assertTrue(np.all(should_be_A.indices == A.indices))
        self.assertTrue(np.all(should_be_A.indptr == A.indptr))

    def test_csr_to_csc_pyomo_exprs(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        A = _CSRMatrix(
            ([5, 8 * m.x, 3 * m.x * m.y**2, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4]), [4, 4]
        )
        thing = A.tocsc()

        self.assertEqual(thing.data[0], 5)
        assertExpressionsEqual(self, thing.data[1], 8 * m.x)
        self.assertEqual(thing.data[2], 6)
        assertExpressionsEqual(self, thing.data[3], 3 * m.x * m.y**2)
        self.assertEqual(thing.data.shape, (4,))

        self.assertTrue(np.all(thing.indices == np.array([0, 1, 3, 2])))
        self.assertTrue(np.all(thing.indptr == np.array([0, 1, 3, 4, 4])))

        should_be_A = thing.tocsr()
        self.assertEqual(should_be_A.data[0], 5)
        assertExpressionsEqual(self, should_be_A.data[1], 8 * m.x)
        assertExpressionsEqual(self, should_be_A.data[2], 3 * m.x * m.y**2)
        self.assertEqual(should_be_A.data[3], 6)
        self.assertEqual(should_be_A.data.shape, (4,))
        self.assertTrue(np.all(should_be_A.indices == np.array([0, 1, 2, 1])))
        self.assertTrue(np.all(should_be_A.indptr == np.array([0, 1, 2, 3, 4])))

    def test_csr_to_csc_empty_matrix(self):
        A = _CSRMatrix(([], [], [0]), [0, 4])
        thing = A.tocsc()

        self.assertEqual(thing.data.size, 0)
        self.assertEqual(thing.indices.size, 0)
        self.assertEqual(thing.shape, (0, 4))
        self.assertTrue(np.all(thing.indptr == np.zeros(5)))

        should_be_A = thing.tocsr()
        self.assertEqual(should_be_A.data.size, 0)
        self.assertEqual(should_be_A.indices.size, 0)
        self.assertEqual(should_be_A.shape, (0, 4))
        self.assertTrue(np.all(should_be_A.indptr == np.zeros(5)))

    def test_todense(self):
        A = _CSRMatrix(([5, 8, 3, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4]), [4, 4])
        dense = np.array([[5, 0, 0, 0], [0, 8, 0, 0], [0, 0, 3, 0], [0, 6, 0, 0]])

        self.assertTrue(np.all(A.todense() == dense))
        self.assertTrue(np.all(A.tocsc().todense() == dense))

        A = _CSRMatrix(
            ([5, 6, 7, 2, 1, 1.5], [0, 1, 1, 2, 3, 1], [0, 2, 4, 5, 6]), [4, 4]
        )
        dense = np.array([[5, 6, 0, 0], [0, 7, 2, 0], [0, 0, 0, 1], [0, 1.5, 0, 0]])
        self.assertTrue(np.all(A.todense() == dense))
        self.assertTrue(np.all(A.tocsc().todense() == dense))

    def test_sum_duplicates(self):
        A = _CSCMatrix(([4, 5], [1, 1], [0, 0, 2, 2]), [3, 3])
        self.assertTrue(np.all(A.data == [4, 5]))
        self.assertTrue(np.all(A.indptr == [0, 0, 2, 2]))
        self.assertTrue(np.all(A.indices == [1, 1]))

        A.sum_duplicates()

        self.assertTrue(np.all(A.data == [9]))
        self.assertTrue(np.all(A.indptr == [0, 0, 1, 1]))
        self.assertTrue(np.all(A.indices == [1]))

        dense = np.array([[0, 0, 0], [0, 9, 0], [0, 0, 0]])
        self.assertTrue(np.all(A.todense() == dense))

    def test_invalid_sparse_matrix_input(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Shape specifies the number of rows as 3 but the index "
            r"pointer has length 2. The index pointer must have length "
            r"nrows \+ 1: Check the 'shape' and 'matrix_data' arguments.",
        ):
            A = _CSRMatrix(([4, 5], [1, 1], [1, 1]), shape=(3, 3))

        with self.assertRaisesRegex(
            ValueError,
            r"Shape specifies the number of columns as 3 but the index "
            r"pointer has length 2. The index pointer must have length "
            r"ncols \+ 1: Check the 'shape' and 'matrix_data' arguments.",
        ):
            A = _CSCMatrix(([4, 5], [1, 1], [1, 1]), shape=(3, 3))


def assertExpressionArraysEqual(self, A, B):
    self.assertEqual(A.shape, B.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assertExpressionsStructurallyEqual(self, A[i, j], B[i, j])


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
        # m.c gets interpreted as a <= Constraint, and you can't really blame
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

    def make_model(self, do_not_flip_c=False):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([0, 1, 3], bounds=lambda m, i: (-1 * (i % 2) * 5, 10 - 12 * (i // 2)))
        m.data = Var([1, 2])
        m.more_data = Var()
        if do_not_flip_c:
            # [ESJ: 06/24]: I should have done this sooner, but if you write c
            # this way, it gets interpreted as a >= constraint, which matches
            # the standard_form tests and makes life much easier. Unforuntately
            # I wrote a lot of tests before I thought of this, so I'm leaving in
            # both variations for the moment.
            m.c = Constraint(
                expr=m.data[1] ** 2 * m.x + 2 * m.y[1] - 3 * m.more_data >= 0
            )
        else:
            m.c = Constraint(expr=m.data[1] ** 2 * m.x + 2 * m.y[1] >= 3 * m.more_data)
        m.d = Constraint(expr=m.y[1] + 4 * m.y[3] <= 5 + m.data[2])
        m.e = Constraint(expr=inequality(-2, m.y[0] + 1 + 6 * m.y[1], 7))
        m.f = Constraint(expr=m.x + (m.data[2] + m.data[1] ** 3) * m.y[0] + 2 == 10)
        m.o = Objective([1, 3], rule=lambda m, i: m.x + i * 5 * m.more_data * m.y[i])
        m.o[1].sense = maximize

        return m

    def test_nonnegative_vars(self):
        m = self.make_model()
        col_order = [m.x, m.y[0], m.y[1], m.y[3]]
        repn = ParameterizedLinearStandardFormCompiler().write(
            m, wrt=[m.data, m.more_data], nonnegative_vars=True, column_order=col_order
        )

        # m.c comes back opposite how it does in test_standard_form, but that's
        # not unexpected.
        self.assertEqual(
            repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 1), (m.f, -1)]
        )
        self.assertEqual(
            list(map(str, repn.x)),
            ['_neg_0', '_pos_0', 'y[0]', '_neg_2', '_pos_2', '_neg_3'],
        )
        ref = np.array(
            [
                [
                    NegationExpression((ProductExpression((-1, m.data[1] ** 2)),)),
                    ProductExpression((-1, m.data[1] ** 2)),
                    0,
                    2,
                    -2,
                    0,
                ],
                [0, 0, 0, -1, 1, -4],
                [0, 0, 1, -6, 6, 0],
                [0, 0, -1, 6, -6, 0],
                [-1, 1, m.data[2] + m.data[1] ** 3, 0, 0, 0],
                [1, -1, -(m.data[2] + m.data[1] ** 3), 0, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.A.todense(), ref)
        assertExpressionListsEqual(
            self,
            repn.b,
            [
                -3 * m.more_data,
                NegationExpression((ProductExpression((-1, 5 + m.data[2])),)),
                6,
                3,
                8,
                -8,
            ],
        )

        c_ref = np.array(
            [
                [1, -1, 0, 5 * m.more_data, -5 * m.more_data, 0],
                [-1, 1, 0, 0, 0, -15 * m.more_data],
            ]
        )
        assertExpressionArraysEqual(self, repn.c.todense(), c_ref)

    def test_slack_form(self):
        m = self.make_model()
        col_order = [m.x, m.y[0], m.y[1], m.y[3]]
        repn = ParameterizedLinearStandardFormCompiler().write(
            m, wrt=[m.data, m.more_data], slack_form=True, column_order=col_order
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            ['x', 'y[0]', 'y[1]', 'y[3]', '_slack_0', '_slack_1', '_slack_2'],
        )
        # m.c is flipped again, so the bounds on _slack_0 are flipped
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [(None, None), (0, 10), (-5, 10), (-5, -2), (0, None), (0, None), (-9, 0)],
        )
        ref = np.array(
            [
                [ProductExpression((-1, m.data[1] ** 2)), 0, -2, 0, 1, 0, 0],
                [0, 0, 1, 4, 0, 1, 0],
                [0, 1, 6, 0, 0, 0, 1],
                [1, m.data[2] + m.data[1] ** 3, 0, 0, 0, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.A.todense(), ref)
        assertExpressionListsEqual(
            self,
            repn.b,
            np.array(
                [
                    -3 * m.more_data,
                    NegationExpression((ProductExpression((-1, 5 + m.data[2])),)),
                    -3,
                    8,
                ]
            ),
        )
        c_ref = np.array(
            [
                [-1, 0, -5 * m.more_data, 0, 0, 0, 0],
                [1, 0, 0, 15 * m.more_data, 0, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.c.todense(), c_ref)

    def test_mixed_form(self):
        m = self.make_model()
        col_order = [m.x, m.y[0], m.y[1], m.y[3]]
        repn = ParameterizedLinearStandardFormCompiler().write(
            m, wrt=[m.data, m.more_data], mixed_form=True, column_order=col_order
        )

        # m.c gets is opposite again
        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 0)])
        self.assertEqual(list(map(str, repn.x)), ['x', 'y[0]', 'y[1]', 'y[3]'])
        self.assertEqual(
            list(v.bounds for v in repn.x), [(None, None), (0, 10), (-5, 10), (-5, -2)]
        )
        ref = np.array(
            [
                [ProductExpression((-1, m.data[1] ** 2)), 0, -2, 0],
                [0, 0, 1, 4],
                [0, 1, 6, 0],
                [0, 1, 6, 0],
                [1, m.data[2] + m.data[1] ** 3, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.A.todense(), ref)
        assertExpressionListsEqual(
            self,
            repn.b,
            np.array(
                [
                    -3 * m.more_data,
                    NegationExpression((ProductExpression((-1, 5 + m.data[2])),)),
                    6,
                    -3,
                    8,
                ]
            ),
        )
        ref_c = np.array([[-1, 0, -5 * m.more_data, 0], [1, 0, 0, 15 * m.more_data]])
        assertExpressionArraysEqual(self, repn.c.todense(), ref_c)

    def test_slack_form_nonnegative_vars(self):
        m = self.make_model(do_not_flip_c=True)
        col_order = [m.x, m.y[0], m.y[1], m.y[3]]
        repn = ParameterizedLinearStandardFormCompiler().write(
            m,
            wrt=[m.data, m.more_data],
            slack_form=True,
            nonnegative_vars=True,
            column_order=col_order,
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            [
                '_neg_0',
                '_pos_0',
                'y[0]',
                '_neg_2',
                '_pos_2',
                '_neg_3',
                '_neg_4',
                '_slack_1',
                '_neg_6',
            ],
        )
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [
                (0, None),
                (0, None),
                (0, 10),
                (0, 5),
                (0, 10),
                (2, 5),
                (0, None),
                (0, None),
                (0, 9),
            ],
        )
        ref = np.array(
            [
                [-m.data[1] ** 2, m.data[1] ** 2, 0, -2, 2, 0, -1, 0, 0],
                [0, 0, 0, -1, 1, -4, 0, 1, 0],
                [0, 0, 1, -6, 6, 0, 0, 0, -1],
                [-1, 1, m.data[2] + m.data[1] ** 3, 0, 0, 0, 0, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.A.todense(), ref)
        assertExpressionListsEqual(
            self,
            repn.b,
            np.array(
                [
                    3 * m.more_data,
                    NegationExpression((ProductExpression((-1, 5 + m.data[2])),)),
                    -3,
                    8,
                ]
            ),
        )
        c_ref = np.array(
            [
                [1, -1, 0, 5 * m.more_data, -5 * m.more_data, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, -15 * m.more_data, 0, 0, 0],
            ]
        )
        assertExpressionArraysEqual(self, repn.c.todense(), c_ref)
