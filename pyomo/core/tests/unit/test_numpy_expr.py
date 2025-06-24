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
import pyomo.common.unittest as unittest

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)

from pyomo.environ import (
    ConcreteModel,
    Var,
    RangeSet,
    Param,
    Objective,
    Set,
    Constraint,
    Reals,
)
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression

from pyomo.repn import generate_standard_repn


@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestNumPy(unittest.TestCase):
    def test_numpy_scalar_times_scalar_var(self):
        # Test issue #685
        m = ConcreteModel()
        m.x = Var()
        e = np.float64(5) * m.x
        self.assertIs(type(e), MonomialTermExpression)
        self.assertTrue(compare_expressions(e, 5.0 * m.x))

        e = m.x * np.float64(5)
        self.assertIs(type(e), MonomialTermExpression)
        self.assertTrue(compare_expressions(e, 5.0 * m.x))

    def test_initialize_param_from_ndarray(self):
        # Test issue #611
        samples = 10
        c1 = 0.5
        c2 = 0.5

        model = ConcreteModel()
        model.i = RangeSet(samples)

        def init_x(model, i):
            return np.random.rand(1)

        def init_y(model, i):
            return c1 * (model.x[i] ** 2) + c2 * model.x[i]

        model.x = Param(model.i, initialize=init_x)
        model.y = Param(model.i, initialize=init_y, domain=Reals)
        model.c_1 = Var(initialize=1)
        model.c_2 = Var(initialize=1)
        model.error = Objective(
            # Sum squared error of quadratic fit
            expr=sum(
                (model.c_1 * model.x[i] ** 2 + model.c_2 * model.x[i] - model.y[i]) ** 2
                for i in model.i
            )
        )
        # model.pprint()

        repn = generate_standard_repn(model.error.expr, compute_values=True)
        self.assertIsNone(repn.nonlinear_expr)
        self.assertEqual(len(repn.quadratic_vars), 3)
        for i in range(3):
            self.assertGreater(repn.quadratic_coefs[i], 0)
        self.assertEqual(len(repn.linear_vars), 2)
        for i in range(2):
            self.assertLess(repn.linear_coefs[i], 0)
        self.assertGreater(repn.constant, 0)

    def test_create_objective_from_numpy(self):
        # Test issue #87
        model = ConcreteModel()

        nsample = 3
        nvariables = 2
        X0 = np.array(range(nsample)).reshape([nsample, 1])
        model.X = 1 + np.array(range(nsample * nvariables)).reshape(
            (nsample, nvariables)
        )
        X = np.concatenate([X0, model.X], axis=1)

        model.I = RangeSet(1, nsample)
        model.J = RangeSet(1, nvariables)

        error = np.ones((nsample, 1))
        beta = np.ones((nvariables + 1, 1))
        model.Y = np.dot(X, beta) + error

        model.beta = Var(model.J)
        model.beta0 = Var()

        def obj_fun(model):
            return sum(
                abs(
                    model.Y[i - 1]
                    - (
                        model.beta0
                        + sum(model.X[i - 1, j - 1] * model.beta[j] for j in model.J)
                    )
                )
                for i in model.I
            )

        model.OBJ = Objective(rule=obj_fun)

        def obj_fun_quad(model):
            return sum(
                (
                    model.Y[i - 1]
                    - (
                        model.beta0
                        + sum(model.X[i - 1, j - 1] * model.beta[j] for j in model.J)
                    )
                )
                ** 2
                for i in model.I
            )

        model.OBJ_QUAD = Objective(rule=obj_fun_quad)

        self.assertEqual(
            str(model.OBJ.expr),
            "abs(4.0 - (beta[1] + 2*beta[2] + beta0)) + "
            "abs(9.0 - (3*beta[1] + 4*beta[2] + beta0)) + "
            "abs(14.0 - (5*beta[1] + 6*beta[2] + beta0))",
        )
        self.assertEqual(model.OBJ.expr.polynomial_degree(), None)
        self.assertEqual(model.OBJ_QUAD.expr.polynomial_degree(), 2)

    @unittest.skipUnless(pandas_available, "pandas is not available")
    def test_param_from_pandas(self):
        # Test issue #68
        model = ConcreteModel()
        model.I = Set(initialize=range(6))

        model.P0 = Param(
            model.I, initialize={0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0}
        )
        model.P1 = Param(
            model.I,
            initialize=pd.Series(
                {0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0}
            ).to_dict(),
        )
        model.P2 = Param(
            model.I,
            initialize=pd.Series({0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0}),
        )

        # model.pprint()
        self.assertEqual(list(model.P0.values()), list(model.P1.values()))
        self.assertEqual(list(model.P0.values()), list(model.P2.values()))

        model.V = Var(model.I, initialize=0)

        def rule(m, l):
            return -m.P0[l] <= m.V[l]

        model.Constraint0 = Constraint(model.I, rule=rule)

        def rule(m, l):
            return -m.P1[l] <= m.V[l]

        model.Constraint1 = Constraint(model.I, rule=rule)

        def rule(m, l):
            return -m.P2[l] <= m.V[l]

        model.Constraint2 = Constraint(model.I, rule=rule)

        # TODO: support vector operations between Indexed objects
        # model.Constraint0a = Constraint(model.I, rule=model.P0 <= model.V)
        # model.Constraint1a = Constraint(model.I, rule=model.P1 <= model.V)
        # model.Constraint2a = Constraint(model.I, rule=model.P2 <= model.V)

    @unittest.skipUnless(pandas_available, "pandas is not available")
    def test_param_from_pandas_series_index(self):
        m = ConcreteModel()
        s = pd.Series([1, 3, 5], index=['T1', 'T2', 'T3'])

        # Params treat Series as maps (so the Series index matters)
        m.I = Set(initialize=s.index)
        m.p1 = Param(m.I, initialize=s)
        self.assertEqual(m.p1.extract_values(), {'T1': 1, 'T2': 3, 'T3': 5})
        m.p2 = Param(s.index, initialize=s)
        self.assertEqual(m.p2.extract_values(), {'T1': 1, 'T2': 3, 'T3': 5})
        with self.assertRaisesRegex(
            KeyError, "Index 'T1' is not valid for indexed component 'p3'"
        ):
            m.p3 = Param([0, 1, 2], initialize=s)

        # Sets treat Series as lists
        m.J = Set(initialize=s)
        self.assertEqual(set(m.J), {1, 3, 5})

    def test_numpy_float(self):
        # Test issue #31
        m = ConcreteModel()

        m.T = Set(initialize=range(3))
        m.v = Var(initialize=1, bounds=(0, None))
        m.c = Var(m.T, initialize=20)
        h = [np.float32(1.0), 1.0, 1]

        def rule(m, t):
            return m.c[0] == h[t] * m.c[0]

        m.x = Constraint(m.T, rule=rule)

        def rule(m, t):
            return m.c[0] == h[t] * m.c[0] * m.v

        m.y = Constraint(m.T, rule=rule)

        def rule(m, t):
            return m.c[0] == h[t] * m.v

        m.z = Constraint(m.T, rule=rule)

        # m.pprint()
        for t in m.T:
            self.assertTrue(compare_expressions(m.x[0].expr, m.x[t].expr))
            self.assertTrue(compare_expressions(m.y[0].expr, m.y[t].expr))
            self.assertTrue(compare_expressions(m.z[0].expr, m.z[t].expr))

    def test_indexed_constraint(self):
        m = ConcreteModel()
        m.x = Var([0, 1, 2, 3])
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([10, 20])
        m.c = Constraint([0, 1], expr=A @ m.x <= b)
        self.assertTrue(
            compare_expressions(
                m.c[0].expr, m.x[0] + 2 * m.x[1] + 3 * m.x[2] + 4 * m.x[3] <= 10
            )
        )
        self.assertTrue(
            compare_expressions(
                m.c[1].expr, 5 * m.x[0] + 6 * m.x[1] + 7 * m.x[2] + 8 * m.x[3] <= 20
            )
        )

    def test_numpy_array_copy_errors(self):
        # Defer testing until here to avoid the unconditional numpy dereference
        if int(np.__version__.split('.')[0]) < 2:
            self.skipTest("requires numpy>=2")

        m = ConcreteModel()
        m.x = Var([0, 1, 2])
        with self.assertRaisesRegex(
            ValueError,
            "Pyomo IndexedComponents do not support conversion to NumPy "
            "arrays without generating a new array",
        ):
            np.asarray(m.x, copy=False)

    def test_numpy_array_dtype_errors(self):
        m = ConcreteModel()
        m.x = Var([0, 1, 2])
        # object is OK
        a = np.asarray(m.x, object)
        self.assertEqual(a.shape, (3,))
        # None is OK
        a = np.asarray(m.x, None)
        self.assertEqual(a.shape, (3,))
        # Anything else is an error
        with self.assertRaisesRegex(
            ValueError,
            "Pyomo IndexedComponents can only be converted to NumPy arrays "
            r"with dtype=object \(received dtype=.*int32",
        ):
            a = np.asarray(m.x, np.int32)

    def test_init_param_from_ndarray(self):
        # Test issue #2033
        m = ConcreteModel()
        m.ix_set = RangeSet(2)

        p_init = np.array([0, 5])

        def init_workaround(model, i):
            return p_init[i - 1]

        m.p = Param(m.ix_set, initialize=init_workaround)
        m.v = Var(m.ix_set)
        expr = m.p[1] > m.v[1]
        self.assertIsInstance(expr, InequalityExpression)
        self.assertEqual(str(expr), "v[1]  <  0")
        expr = m.p[2] > m.v[2]
        self.assertIsInstance(expr, InequalityExpression)
        self.assertEqual(str(expr), "v[2]  <  5")


if __name__ == '__main__':
    unittest.main()
