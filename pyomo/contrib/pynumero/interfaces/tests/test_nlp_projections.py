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
import os

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run NLP tests")

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
    RenamedNLP,
    ProjectedNLP,
    ProjectedExtendedNLP,
)


def create_pyomo_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=(-10, 10), initialize={0: 1.0, 1: 2.0, 2: 4.0})

    m.obj = pyo.Objective(
        expr=m.x[0] ** 2 + m.x[0] * m.x[1] + m.x[0] * m.x[2] + m.x[2] ** 2
    )

    m.con1 = pyo.Constraint(expr=m.x[0] * m.x[1] + m.x[0] * m.x[2] == 4)
    m.con2 = pyo.Constraint(expr=m.x[0] + m.x[2] == 4)

    return m


class TestRenamedNLP(unittest.TestCase):
    def test_rename(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(nlp.primals_names(), expected_names)
        renamed_nlp = RenamedNLP(nlp, {'x[0]': 'y[0]', 'x[1]': 'y[1]', 'x[2]': 'y[2]'})
        expected_names = ['y[0]', 'y[1]', 'y[2]']


class TestProjectedNLP(unittest.TestCase):
    def test_projected(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(
            np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 2.0, 4.0]))
        )
        self.assertTrue(
            np.array_equal(
                projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 1.0, 9.0])
            )
        )
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test a reordering
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(
            np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0, 2.0]))
        )
        self.assertTrue(
            np.array_equal(
                projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 9.0, 1.0])
            )
        )
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'y', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'y', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(
            projected_nlp.get_primals(), np.asarray([1.0, 4.0, np.nan, 2.0])
        )

        self.assertTrue(
            np.array_equal(
                projected_nlp.evaluate_grad_objective(),
                np.asarray([8.0, 9.0, 0.0, 1.0]),
            )
        )
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray(
            [
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]'])
        expected_names = ['x[0]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0]))

        self.assertTrue(
            np.array_equal(
                projected_nlp.evaluate_grad_objective(), np.asarray([8.0, 9.0])
            )
        )
        self.assertEqual(projected_nlp.nnz_jacobian(), 4)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 4)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 4)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0 * J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 4)
        expectedH = np.asarray([[2.0, 1.0], [1.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0 * H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))


class TestProjectedExtendedNLP(unittest.TestCase):
    def _make_model_with_inequalities(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=range(4))
        m.x = pyo.Var(m.I, initialize=1.1)
        m.obj = pyo.Objective(
            expr=1 * m.x[0] + 2 * m.x[1] ** 2 + 3 * m.x[1] * m.x[2] + 4 * m.x[3] ** 3
        )
        m.eq_con_1 = pyo.Constraint(
            expr=m.x[0] * (m.x[1] ** 1.1) * (m.x[2] ** 1.2) == 3.0
        )
        m.eq_con_2 = pyo.Constraint(expr=m.x[0] ** 2 + m.x[3] ** 2 + m.x[1] == 2.0)
        m.ineq_con_1 = pyo.Constraint(expr=m.x[0] + m.x[3] * m.x[0] <= 4.0)
        m.ineq_con_2 = pyo.Constraint(expr=m.x[1] + m.x[2] >= 1.0)
        m.ineq_con_3 = pyo.Constraint(expr=m.x[2] >= 0)
        return m

    def _get_nlps(self):
        m = self._make_model_with_inequalities()
        nlp = PyomoNLP(m)
        primals_ordering = ["x[1]", "x[0]"]
        proj_nlp = ProjectedExtendedNLP(nlp, primals_ordering)
        return m, nlp, proj_nlp

    def _x_to_nlp(self, m, nlp, values):
        # We often want to set coordinates in the nlp based on some
        # order of variables in the model. However, in general we don't
        # know the order of primals in the NLP. This method reorders
        # a list of values such that they will be sent to x[0]...x[3]
        # in the NLP.
        indices = nlp.get_primal_indices([m.x[0], m.x[1], m.x[2], m.x[3]])
        reordered_values = [None for _ in m.x]
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _c_to_nlp(self, m, nlp, values):
        indices = nlp.get_constraint_indices(
            [m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3]
        )
        reordered_values = [None] * 5
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _eq_to_nlp(self, m, nlp, values):
        indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
        reordered_values = [None] * 2
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _ineq_to_nlp(self, m, nlp, values):
        indices = nlp.get_inequality_constraint_indices(
            [m.ineq_con_1, m.ineq_con_2, m.ineq_con_3]
        )
        reordered_values = [None] * 3
        for i, val in zip(indices, values):
            reordered_values[i] = val
        return reordered_values

    def _rc_to_nlp(self, m, nlp, rc):
        var_indices = nlp.get_primal_indices(list(m.x.values()))
        con_indices = nlp.get_constraint_indices(
            [m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3]
        )
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp(self, m, nlp, rc):
        var_indices = [1, 0]
        con_indices = nlp.get_constraint_indices(
            [m.eq_con_1, m.eq_con_2, m.ineq_con_1, m.ineq_con_2, m.ineq_con_3]
        )
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp_eq(self, m, nlp, rc):
        # Expects variable coords in order [x0, x1], constraint coords
        # in order [eq1, eq2]
        var_indices = [1, 0]
        con_indices = nlp.get_equality_constraint_indices([m.eq_con_1, m.eq_con_2])
        i, j = rc
        return (con_indices[i], var_indices[j])

    def _rc_to_proj_nlp_ineq(self, m, nlp, rc):
        # Expects variable coords in order [x0, x1], constraint coords
        # in order [ineq1, ineq2, ineq3]
        var_indices = [1, 0]
        con_indices = nlp.get_inequality_constraint_indices(
            [m.ineq_con_1, m.ineq_con_2, m.ineq_con_3]
        )
        i, j = rc
        return (con_indices[i], var_indices[j])

    def test_non_extended_original_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        # Note that even though nlp is a PyomoNLP, and thus an ExtendedNLP,
        # this projected NLP is *not* extended.
        proj_nlp = ProjectedNLP(nlp, ["x[0]", "x[1]", "x[2]"])
        msg = "Original NLP must be an instance of ExtendedNLP"
        with self.assertRaisesRegex(TypeError, msg):
            proj_ext_nlp = ProjectedExtendedNLP(proj_nlp, ["x[1]", "x[0]"])

    def test_n_primals_constraints(self):
        m, nlp, proj_nlp = self._get_nlps()
        self.assertEqual(proj_nlp.n_primals(), 2)
        self.assertEqual(proj_nlp.n_constraints(), 5)
        self.assertEqual(proj_nlp.n_eq_constraints(), 2)
        self.assertEqual(proj_nlp.n_ineq_constraints(), 3)

    def test_set_get_primals(self):
        m, nlp, proj_nlp = self._get_nlps()
        primals = proj_nlp.get_primals()
        np.testing.assert_array_equal(primals, [1.1, 1.1])
        nlp.set_primals(self._x_to_nlp(m, nlp, [1.2, 1.3, 1.4, 1.5]))
        proj_primals = proj_nlp.get_primals()
        np.testing.assert_array_equal(primals, [1.3, 1.2])

        proj_nlp.set_primals(np.array([-1.0, -1.1]))
        # Make sure we can get this vector back from ProjNLP
        np.testing.assert_array_equal(proj_nlp.get_primals(), [-1.0, -1.1])
        # Make sure we can get this vector back from the original NLP
        np.testing.assert_array_equal(
            nlp.get_primals(), self._x_to_nlp(m, nlp, [-1.1, -1.0, 1.4, 1.5])
        )

    def test_set_primals_with_list_error(self):
        # This doesn't work. Get a TypeError due to treating list as numpy array
        # when indexing another array.
        m, nlp, proj_nlp = self._get_nlps()
        msg = "only integer scalar arrays can be converted to a scalar index"
        # This test may be too specific. If NumPy changes this error message,
        # the test could fail.
        with self.assertRaisesRegex(TypeError, msg):
            proj_nlp.set_primals([1.0, 2.0])

    def test_get_set_duals(self):
        m, nlp, proj_nlp = self._get_nlps()
        nlp.set_duals([2, 3, 4, 5, 6])
        np.testing.assert_array_equal(proj_nlp.get_duals(), [2, 3, 4, 5, 6])

        proj_nlp.set_duals([-1, -2, -3, -4, -5])
        np.testing.assert_array_equal(proj_nlp.get_duals(), [-1, -2, -3, -4, -5])
        np.testing.assert_array_equal(nlp.get_duals(), [-1, -2, -3, -4, -5])

    def test_eval_constraints(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        con_resids = nlp.evaluate_constraints()
        pred_con_body = [
            x0 * x1**1.1 * x2**1.2 - 3.0,
            x0**2 + x3**2 + x1 - 2.0,
            x0 + x0 * x3,
            x1 + x2,
            x2,
        ]
        np.testing.assert_array_equal(con_resids, self._c_to_nlp(m, nlp, pred_con_body))

        con_resids = proj_nlp.evaluate_constraints()
        np.testing.assert_array_equal(con_resids, self._c_to_nlp(m, nlp, pred_con_body))

        eq_resids = proj_nlp.evaluate_eq_constraints()
        pred_eq_body = [x0 * x1**1.1 * x2**1.2 - 3.0, x0**2 + x3**2 + x1 - 2.0]
        np.testing.assert_array_equal(eq_resids, self._eq_to_nlp(m, nlp, pred_eq_body))

        ineq_body = proj_nlp.evaluate_ineq_constraints()
        pred_ineq_body = [x0 + x0 * x3, x1 + x2, x2]
        np.testing.assert_array_equal(
            ineq_body, self._ineq_to_nlp(m, nlp, pred_ineq_body)
        )

    def test_eval_jacobian_orig_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        jac = nlp.evaluate_jacobian()
        # Predicted row/col indices in the "natural ordering" of the model
        pred_rc = [
            # eq 1
            (0, 0),
            (0, 1),
            (0, 2),
            # eq 2
            (1, 0),
            (1, 1),
            (1, 3),
            # ineq 1
            (2, 0),
            (2, 3),
            # ineq 2
            (3, 1),
            (3, 2),
            # ineq 3
            (4, 2),
        ]
        pred_data_dict = {
            # eq 1
            (0, 0): x1**1.1 * x2**1.2,
            (0, 1): 1.1 * x0 * (x1**0.1) * x2**1.2,
            (0, 2): 1.2 * x0 * x1**1.1 * x2**0.2,
            # eq 2
            (1, 0): 2 * x0,
            (1, 1): 1.0,
            (1, 3): 2 * x3,
            # ineq 1
            (2, 0): 1.0 + x3,
            (2, 3): x0,
            # ineq 2
            (3, 1): 1.0,
            (3, 2): 1.0,
            # ineq 3
            (4, 2): 1.0,
        }
        pred_rc_set = set(self._rc_to_nlp(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_nlp(m, nlp, rc): val for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        jac = proj_nlp.evaluate_jacobian()
        self.assertEqual(jac.shape, (5, 2))
        # Predicted row/col indices. In the "natural ordering" of the model.
        pred_rc = [
            # eq 1
            (0, 0),
            (0, 1),
            # eq 2
            (1, 0),
            (1, 1),
            # ineq 1
            (2, 0),
            # ineq 2
            (3, 1),
        ]
        pred_data_dict = {
            # eq 1
            (0, 0): x1**1.1 * x2**1.2,
            (0, 1): 1.1 * x0 * (x1**0.1) * x2**1.2,
            # eq 2
            (1, 0): 2 * x0,
            (1, 1): 1.0,
            # ineq 1
            (2, 0): 1.0 + x3,
            # ineq 2
            (3, 1): 1.0,
        }
        # Projected NLP has primals: [x1, x0]
        pred_rc_set = set(self._rc_to_proj_nlp(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_proj_nlp(m, nlp, rc): val for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_eq_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        jac = proj_nlp.evaluate_jacobian_eq()
        self.assertEqual(jac.shape, (2, 2))
        # Predicted row/col indices. In the "natural ordering" of the equality
        # constraints (eq1, eq2)
        # In list, first two are eq 1; second two are eq 2
        pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pred_data_dict = {
            # eq 1
            (0, 0): x1**1.1 * x2**1.2,
            (0, 1): 1.1 * x0 * (x1**0.1) * x2**1.2,
            # eq 2
            (1, 0): 2 * x0,
            (1, 1): 1.0,
        }
        # Projected NLP has primals: [x1, x0]
        pred_rc_set = set(self._rc_to_proj_nlp_eq(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_proj_nlp_eq(m, nlp, rc): val
            for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_ineq_jacobian_proj_nlp(self):
        m, nlp, proj_nlp = self._get_nlps()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        jac = proj_nlp.evaluate_jacobian_ineq()
        self.assertEqual(jac.shape, (3, 2))
        # Predicted row/col indices. In the "natural ordering" of the inequality
        # constraints (ineq1, ineq2, ineq3)
        pred_rc = [(0, 0), (1, 1)]  # [(ineq 1, ineq 2)]
        pred_data_dict = {
            # ineq 1
            (0, 0): 1.0 + x3,
            # ineq 2
            (1, 1): 1.0,
        }
        # Projected NLP has primals: [x1, x0]
        pred_rc_set = set(self._rc_to_proj_nlp_ineq(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_proj_nlp_ineq(m, nlp, rc): val
            for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_eq_jacobian_proj_nlp_using_out_arg(self):
        m, nlp, proj_nlp = self._get_nlps()
        jac = proj_nlp.evaluate_jacobian_eq()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        proj_nlp.evaluate_jacobian_eq(out=jac)
        self.assertEqual(jac.shape, (2, 2))
        # Predicted row/col indices. In the "natural ordering" of the equality
        # constraints (eq1, eq2)
        # In list, first two are eq 1; second two are eq 2
        pred_rc = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pred_data_dict = {
            # eq 1
            (0, 0): x1**1.1 * x2**1.2,
            (0, 1): 1.1 * x0 * (x1**0.1) * x2**1.2,
            # eq 2
            (1, 0): 2 * x0,
            (1, 1): 1.0,
        }
        # Projected NLP has primals: [x1, x0]
        pred_rc_set = set(self._rc_to_proj_nlp_eq(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_proj_nlp_eq(m, nlp, rc): val
            for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)

    def test_eval_ineq_jacobian_proj_nlp_using_out_arg(self):
        m, nlp, proj_nlp = self._get_nlps()
        jac = proj_nlp.evaluate_jacobian_ineq()
        x0, x1, x2, x3 = [1.2, 1.3, 1.4, 1.5]
        nlp.set_primals(self._x_to_nlp(m, nlp, [x0, x1, x2, x3]))

        proj_nlp.evaluate_jacobian_ineq(out=jac)
        self.assertEqual(jac.shape, (3, 2))
        # Predicted row/col indices. In the "natural ordering" of the inequality
        # constraints (ineq1, ineq2, ineq3)
        pred_rc = [(0, 0), (1, 1)]  # [(ineq 1, ineq 2)]
        pred_data_dict = {
            # ineq 1
            (0, 0): 1.0 + x3,
            # ineq 2
            (1, 1): 1.0,
        }
        # Projected NLP has primals: [x1, x0]
        pred_rc_set = set(self._rc_to_proj_nlp_ineq(m, nlp, rc) for rc in pred_rc)
        pred_data_dict = {
            self._rc_to_proj_nlp_ineq(m, nlp, rc): val
            for rc, val in pred_data_dict.items()
        }
        rc_set = set(zip(jac.row, jac.col))
        self.assertEqual(pred_rc_set, rc_set)

        data_dict = dict(zip(zip(jac.row, jac.col), jac.data))
        self.assertEqual(pred_data_dict, data_dict)


if __name__ == '__main__':
    TestRenamedNLP().test_rename()
    TestProjectedNLP().test_projected()
