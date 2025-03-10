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

import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import math

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.dependencies.scipy import sparse as spa

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests"
    )

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
    CyIpoptNLP,
    cyipopt_available,
)

from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP, PyomoNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)

from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
    check_vectors_specific_order,
    check_sparse_matrix_specific_order,
)


def create_pyomo_model(A1, A2, c1, c2, N, dt):
    m = pyo.ConcreteModel()

    # timesteps
    m.T = pyo.Set(initialize=list(range(N)), ordered=True)
    m.Tu = pyo.Set(initialize=list(range(N))[1:], ordered=True)

    # inputs (controls)
    m.F1 = pyo.Var(m.Tu, bounds=(0, None), initialize={t: 1 + 0.1 * t for t in m.Tu})
    m.F2 = pyo.Var(m.Tu, bounds=(0, None), initialize={t: 2 + 0.1 * t for t in m.Tu})

    # state variables
    m.h1 = pyo.Var(m.T, bounds=(0, None), initialize={t: 3 + 0.1 * t for t in m.T})
    m.h2 = pyo.Var(m.T, bounds=(0, None), initialize={t: 4 + 0.1 * t for t in m.T})

    # algebraics (outputs)
    m.F12 = pyo.Var(m.T, bounds=(0, None), initialize={t: 5 + 0.1 * t for t in m.T})
    m.Fo = pyo.Var(m.T, bounds=(0, None), initialize={t: 6 + 0.1 * t for t in m.T})

    @m.Constraint(m.Tu)
    def h1bal(m, t):
        return (m.h1[t] - m.h1[t - 1]) - dt / A1 * (
            m.F1[t] - c1 * pyo.sqrt(m.h1[t])
        ) == 0

    @m.Constraint(m.Tu)
    def h2bal(m, t):
        return (m.h2[t] - m.h2[t - 1]) - dt / A2 * (
            c1 * pyo.sqrt(m.h1[t]) + m.F2[t] - c2 * pyo.sqrt(m.h2[t])
        ) == 0

    @m.Constraint(m.T)
    def F12con(m, t):
        return c1 * pyo.sqrt(m.h1[t]) - m.F12[t] == 0

    @m.Constraint(m.T)
    def Focon(m, t):
        return c2 * pyo.sqrt(m.h2[t]) - m.Fo[t] == 0

    @m.Constraint(m.Tu)
    def min_inflow(m, t):
        return 2 <= m.F1[t]

    @m.Constraint(m.T)
    def max_outflow(m, t):
        return m.Fo[t] <= 4.5

    m.h10 = pyo.Constraint(expr=m.h1[m.T.first()] == 1.5)
    m.h20 = pyo.Constraint(expr=m.h2[m.T.first()] == 0.5)
    m.obj = pyo.Objective(
        expr=sum((m.h1[t] - 1.0) ** 2 + (m.h2[t] - 1.5) ** 2 for t in m.T)
    )

    return m


class TwoTanksSeries(ExternalGreyBoxModel):
    def __init__(self, A1, A2, c1, c2, N, dt):
        self._A1 = A1
        self._A2 = A2
        self._c1 = c1
        self._c2 = c2
        self._N = N
        self._dt = dt
        self._input_names = ['F1_{}'.format(t) for t in range(1, N)]
        self._input_names.extend(['F2_{}'.format(t) for t in range(1, N)])
        self._input_names.extend(['h1_{}'.format(t) for t in range(0, N)])
        self._input_names.extend(['h2_{}'.format(t) for t in range(0, N)])
        self._output_names = ['F12_{}'.format(t) for t in range(0, N)]
        self._output_names.extend(['Fo_{}'.format(t) for t in range(0, N)])
        self._equality_constraint_names = ['h1bal_{}'.format(t) for t in range(1, N)]
        self._equality_constraint_names.extend(
            ['h2bal_{}'.format(t) for t in range(1, N)]
        )

        # inputs
        self._F1 = np.zeros(N)  # we don't use the first one
        self._F2 = np.zeros(N)  # we don't use the first one
        self._h1 = np.zeros(N)
        self._h2 = np.zeros(N)

        # outputs
        self._F12 = np.zeros(N)
        self._Fo = np.zeros(N)

        # multipliers
        self._eq_con_mult_values = np.ones(2 * (N - 1))
        self._output_con_mult_values = np.ones(2 * N)

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def finalize_block_construction(self, pyomo_block):
        # initialize all variables to 1 and set bounds
        for k in pyomo_block.inputs:
            pyomo_block.inputs[k].setlb(0)
            pyomo_block.inputs[k].value = 1.0
        for k in pyomo_block.outputs:
            pyomo_block.outputs[k].setlb(0)
            pyomo_block.outputs[k].value = 1.0

    def set_input_values(self, input_values):
        N = self._N
        assert len(input_values) == 4 * N - 2
        self._F1[1 : self._N] = np.copy(input_values[: N - 1])
        self._F2[1 : self._N] = np.copy(input_values[N - 1 : 2 * N - 2])
        self._h1 = np.copy(input_values[2 * N - 2 : 3 * N - 2])
        self._h2 = np.copy(input_values[3 * N - 2 : 4 * N - 2])

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2 * (self._N - 1)
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 2 * self._N
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2

        resid = np.zeros(2 * (N - 1))

        for t in range(1, N):
            resid[t - 1] = (h1[t] - h1[t - 1]) - self._dt / self._A1 * (
                F1[t] - self._c1 * math.sqrt(h1[t])
            )

        for t in range(1, N):
            resid[t - 2 + N] = (h2[t] - h2[t - 1]) - self._dt / self._A2 * (
                self._c1 * math.sqrt(h1[t]) + F2[t] - self._c2 * math.sqrt(h2[t])
            )
        return resid

    def evaluate_outputs(self):
        N = self._N
        h1 = self._h1
        h2 = self._h2

        resid = np.zeros(2 * N)

        for t in range(N):
            resid[t] = self._c1 * math.sqrt(h1[t])

        for t in range(N):
            resid[t + N] = self._c2 * math.sqrt(h2[t])

        return resid

    def evaluate_jacobian_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt

        nnz = 3 * (N - 1) + 4 * (N - 1)
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        # Jac h1bal
        for i in range(N - 1):
            irow[idx] = i
            jcol[idx] = i
            data[idx] = -dt / A1
            idx += 1
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = -1
            idx += 1
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = 1 + dt / A1 * c1 * 1 / 2 * (h1[i + 1]) ** (-0.5)
            idx += 1
        # Jac h2bal
        for i in range(N - 1):
            irow[idx] = i + (N - 1)
            jcol[idx] = i + (N - 1)
            data[idx] = -dt / A2
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = -dt / A2 * c1 * 1 / 2 * (h1[i + 1]) ** (-0.5)
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = -1
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + N + i + 1
            data[idx] = 1 + dt / A2 * c2 * 1 / 2 * (h2[i + 1]) ** (-0.5)
            idx += 1

        assert idx == nnz
        return spa.coo_matrix(
            (data, (irow, jcol)), shape=(2 * (N - 1), 2 * (N - 1) + 2 * N)
        )

    def evaluate_jacobian_outputs(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt

        nnz = 2 * N
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        # Jac F12
        for i in range(N):
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = 1 / 2 * c1 * h1[i] ** (-0.5)
            idx += 1
        for i in range(N):
            irow[idx] = N + i
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = 1 / 2 * c2 * h2[i] ** (-0.5)
            idx += 1

        assert idx == nnz
        return spa.coo_matrix((data, (irow, jcol)), shape=(2 * N, 2 * (N - 1) + 2 * N))

    def evaluate_hessian_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        lam = self._eq_con_mult_values

        nnz = 2 * (N - 1)
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        for i in range(N - 1):
            irow[idx] = 2 * (N - 1) + i + 1
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = lam[i] * dt / A1 * (-c1 / 4) * h1[i + 1] ** (-1.5) + lam[
                (N - 1) + i
            ] * dt / A2 * (c1 / 4) * h1[i + 1] ** (-1.5)
            idx += 1
            irow[idx] = 2 * (N - 1) + N + i + 1
            jcol[idx] = 2 * (N - 1) + N + i + 1
            data[idx] = lam[(N - 1) + i] * dt / A2 * (-c2 / 4) * h2[i + 1] ** (-1.5)
            idx += 1

        assert idx == nnz
        hess = spa.coo_matrix(
            (data, (irow, jcol)), shape=(2 * (N - 1) + 2 * N, 2 * (N - 1) + 2 * N)
        )
        return hess

    def evaluate_hessian_outputs(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        lam = self._output_con_mult_values

        nnz = 2 * N
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0

        # Hess F12_t
        for i in range(N):
            irow[idx] = 2 * (N - 1) + i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = lam[i] * c1 * (-1 / 4) * h1[i] ** (-1.5)
            idx += 1
        # Hess Fo_t
        for i in range(N):
            irow[idx] = 2 * (N - 1) + N + i
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = lam[N + i] * c2 * (-1 / 4) * h2[i] ** (-1.5)
            idx += 1

        assert idx == nnz
        hess = spa.coo_matrix(
            (data, (irow, jcol)), shape=(2 * (N - 1) + 2 * N, 2 * (N - 1) + 2 * N)
        )
        return hess


def create_pyomo_external_grey_box_model(A1, A2, c1, c2, N, dt):
    m2 = pyo.ConcreteModel()
    m2.T = pyo.Set(initialize=list(range(N)), ordered=True)
    m2.Tu = pyo.Set(initialize=list(range(N))[1:], ordered=True)
    m2.egb = ExternalGreyBoxBlock()
    m2.egb.set_external_model(TwoTanksSeries(A1, A2, c1, c2, N, dt))

    # initialize the same as the pyomo model
    for t in m2.Tu:
        m2.egb.inputs['F1_{}'.format(t)].value = 1 + 0.1 * t
        m2.egb.inputs['F2_{}'.format(t)].value = 2 + 0.1 * t
    for t in m2.T:
        m2.egb.inputs['h1_{}'.format(t)].value = 3 + 0.1 * t
        m2.egb.inputs['h2_{}'.format(t)].value = 4 + 0.1 * t
        m2.egb.outputs['F12_{}'.format(t)].value = 5 + 0.1 * t
        m2.egb.outputs['Fo_{}'.format(t)].value = 6 + 0.1 * t

    @m2.Constraint(m2.Tu)
    def min_inflow(m, t):
        F1_t = m.egb.inputs['F1_{}'.format(t)]
        return 2 <= F1_t

    @m2.Constraint(m2.T)
    def max_outflow(m, t):
        Fo_t = m.egb.outputs['Fo_{}'.format(t)]
        return Fo_t <= 4.5

    m2.h10 = pyo.Constraint(expr=m2.egb.inputs['h1_0'] == 1.5)
    m2.h20 = pyo.Constraint(expr=m2.egb.inputs['h2_0'] == 0.5)
    m2.obj = pyo.Objective(
        expr=sum(
            (m2.egb.inputs['h1_{}'.format(t)] - 1.0) ** 2
            + (m2.egb.inputs['h2_{}'.format(t)] - 1.5) ** 2
            for t in m2.T
        )
    )

    return m2


class TestGreyBoxModel(unittest.TestCase):
    @unittest.skipIf(
        not pyo.SolverFactory('ipopt').available(exception_flag=False),
        "Ipopt needed to run tests with solve",
    )
    def test_compare_evaluations(self):
        A1 = 5
        A2 = 10
        c1 = 3
        c2 = 4
        N = 6
        dt = 1

        m = create_pyomo_model(A1, A2, c1, c2, N, dt)
        solver = pyo.SolverFactory('ipopt')
        solver.options['linear_solver'] = 'mumps'
        status = solver.solve(m, tee=False)
        m_nlp = PyomoNLP(m)

        mex = create_pyomo_external_grey_box_model(A1, A2, c1, c2, N, dt)
        # mex_nlp = PyomoGreyBoxNLP(mex)
        mex_nlp = PyomoNLPWithGreyBoxBlocks(mex)

        # get the variable and constraint order and create the maps
        # reliable order independent comparisons
        m_x_order = m_nlp.primals_names()
        m_c_order = m_nlp.constraint_names()
        mex_x_order = mex_nlp.primals_names()
        mex_c_order = mex_nlp.constraint_names()

        x1list = [
            'h1[0]',
            'h1[1]',
            'h1[2]',
            'h1[3]',
            'h1[4]',
            'h1[5]',
            'h2[0]',
            'h2[1]',
            'h2[2]',
            'h2[3]',
            'h2[4]',
            'h2[5]',
            'F1[1]',
            'F1[2]',
            'F1[3]',
            'F1[4]',
            'F1[5]',
            'F2[1]',
            'F2[2]',
            'F2[3]',
            'F2[4]',
            'F2[5]',
            'F12[0]',
            'F12[1]',
            'F12[2]',
            'F12[3]',
            'F12[4]',
            'F12[5]',
            'Fo[0]',
            'Fo[1]',
            'Fo[2]',
            'Fo[3]',
            'Fo[4]',
            'Fo[5]',
        ]
        x2list = [
            'egb.inputs[h1_0]',
            'egb.inputs[h1_1]',
            'egb.inputs[h1_2]',
            'egb.inputs[h1_3]',
            'egb.inputs[h1_4]',
            'egb.inputs[h1_5]',
            'egb.inputs[h2_0]',
            'egb.inputs[h2_1]',
            'egb.inputs[h2_2]',
            'egb.inputs[h2_3]',
            'egb.inputs[h2_4]',
            'egb.inputs[h2_5]',
            'egb.inputs[F1_1]',
            'egb.inputs[F1_2]',
            'egb.inputs[F1_3]',
            'egb.inputs[F1_4]',
            'egb.inputs[F1_5]',
            'egb.inputs[F2_1]',
            'egb.inputs[F2_2]',
            'egb.inputs[F2_3]',
            'egb.inputs[F2_4]',
            'egb.inputs[F2_5]',
            'egb.outputs[F12_0]',
            'egb.outputs[F12_1]',
            'egb.outputs[F12_2]',
            'egb.outputs[F12_3]',
            'egb.outputs[F12_4]',
            'egb.outputs[F12_5]',
            'egb.outputs[Fo_0]',
            'egb.outputs[Fo_1]',
            'egb.outputs[Fo_2]',
            'egb.outputs[Fo_3]',
            'egb.outputs[Fo_4]',
            'egb.outputs[Fo_5]',
        ]
        x1_x2_map = dict(zip(x1list, x2list))
        x1idx_x2idx_map = {
            i: mex_x_order.index(x1_x2_map[m_x_order[i]]) for i in range(len(m_x_order))
        }

        c1list = [
            'h1bal[1]',
            'h1bal[2]',
            'h1bal[3]',
            'h1bal[4]',
            'h1bal[5]',
            'h2bal[1]',
            'h2bal[2]',
            'h2bal[3]',
            'h2bal[4]',
            'h2bal[5]',
            'F12con[0]',
            'F12con[1]',
            'F12con[2]',
            'F12con[3]',
            'F12con[4]',
            'F12con[5]',
            'Focon[0]',
            'Focon[1]',
            'Focon[2]',
            'Focon[3]',
            'Focon[4]',
            'Focon[5]',
            'min_inflow[1]',
            'min_inflow[2]',
            'min_inflow[3]',
            'min_inflow[4]',
            'min_inflow[5]',
            'max_outflow[0]',
            'max_outflow[1]',
            'max_outflow[2]',
            'max_outflow[3]',
            'max_outflow[4]',
            'max_outflow[5]',
            'h10',
            'h20',
        ]
        c2list = [
            'egb.h1bal_1',
            'egb.h1bal_2',
            'egb.h1bal_3',
            'egb.h1bal_4',
            'egb.h1bal_5',
            'egb.h2bal_1',
            'egb.h2bal_2',
            'egb.h2bal_3',
            'egb.h2bal_4',
            'egb.h2bal_5',
            'egb.output_constraints[F12_0]',
            'egb.output_constraints[F12_1]',
            'egb.output_constraints[F12_2]',
            'egb.output_constraints[F12_3]',
            'egb.output_constraints[F12_4]',
            'egb.output_constraints[F12_5]',
            'egb.output_constraints[Fo_0]',
            'egb.output_constraints[Fo_1]',
            'egb.output_constraints[Fo_2]',
            'egb.output_constraints[Fo_3]',
            'egb.output_constraints[Fo_4]',
            'egb.output_constraints[Fo_5]',
            'min_inflow[1]',
            'min_inflow[2]',
            'min_inflow[3]',
            'min_inflow[4]',
            'min_inflow[5]',
            'max_outflow[0]',
            'max_outflow[1]',
            'max_outflow[2]',
            'max_outflow[3]',
            'max_outflow[4]',
            'max_outflow[5]',
            'h10',
            'h20',
        ]
        c1_c2_map = dict(zip(c1list, c2list))
        c1idx_c2idx_map = {
            i: mex_c_order.index(c1_c2_map[m_c_order[i]]) for i in range(len(m_c_order))
        }

        # get the primals from m and put them in the correct order for mex
        m_x = m_nlp.get_primals()
        mex_x = np.zeros(len(m_x))
        for i in range(len(m_x)):
            mex_x[x1idx_x2idx_map[i]] = m_x[i]

        # get the duals from m and put them in the correct order for mex
        m_lam = m_nlp.get_duals()
        mex_lam = np.zeros(len(m_lam))
        for i in range(len(m_x)):
            mex_lam[c1idx_c2idx_map[i]] = m_lam[i]

        mex_nlp.set_primals(mex_x)
        mex_nlp.set_duals(mex_lam)

        m_obj = m_nlp.evaluate_objective()
        mex_obj = mex_nlp.evaluate_objective()
        self.assertAlmostEqual(m_obj, mex_obj, places=4)

        m_gobj = m_nlp.evaluate_grad_objective()
        mex_gobj = mex_nlp.evaluate_grad_objective()
        check_vectors_specific_order(
            self, m_gobj, m_x_order, mex_gobj, mex_x_order, x1_x2_map
        )

        m_c = m_nlp.evaluate_constraints()
        mex_c = mex_nlp.evaluate_constraints()
        check_vectors_specific_order(
            self, m_c, m_c_order, mex_c, mex_c_order, c1_c2_map
        )

        m_j = m_nlp.evaluate_jacobian()
        mex_j = mex_nlp.evaluate_jacobian().todense()
        check_sparse_matrix_specific_order(
            self,
            m_j,
            m_c_order,
            m_x_order,
            mex_j,
            mex_c_order,
            mex_x_order,
            c1_c2_map,
            x1_x2_map,
        )

        m_h = m_nlp.evaluate_hessian_lag()
        mex_h = mex_nlp.evaluate_hessian_lag()
        check_sparse_matrix_specific_order(
            self,
            m_h,
            m_x_order,
            m_x_order,
            mex_h,
            mex_x_order,
            mex_x_order,
            x1_x2_map,
            x1_x2_map,
        )

        mex_h = 0 * mex_h
        mex_nlp.evaluate_hessian_lag(out=mex_h)
        check_sparse_matrix_specific_order(
            self,
            m_h,
            m_x_order,
            m_x_order,
            mex_h,
            mex_x_order,
            mex_x_order,
            x1_x2_map,
            x1_x2_map,
        )

    @unittest.skipIf(not cyipopt_available, "CyIpopt needed to run tests with solve")
    def test_solve(self):
        A1 = 5
        A2 = 10
        c1 = 3
        c2 = 4
        N = 6
        dt = 1

        m = create_pyomo_model(A1, A2, c1, c2, N, dt)
        solver = pyo.SolverFactory('cyipopt')
        solver.config.options['linear_solver'] = 'mumps'
        status = solver.solve(m, tee=False)

        mex = create_pyomo_external_grey_box_model(A1, A2, c1, c2, N, dt)
        solver = pyo.SolverFactory('cyipopt')
        solver.config.options['linear_solver'] = 'mumps'
        status = solver.solve(mex, tee=False)

        for k in m.F1:
            self.assertAlmostEqual(
                pyo.value(m.F1[k]),
                pyo.value(mex.egb.inputs['F1_{}'.format(k)]),
                places=3,
            )
            self.assertAlmostEqual(
                pyo.value(m.F2[k]),
                pyo.value(mex.egb.inputs['F2_{}'.format(k)]),
                places=3,
            )
        for k in m.h1:
            self.assertAlmostEqual(
                pyo.value(m.h1[k]),
                pyo.value(mex.egb.inputs['h1_{}'.format(k)]),
                places=3,
            )
            self.assertAlmostEqual(
                pyo.value(m.h2[k]),
                pyo.value(mex.egb.inputs['h2_{}'.format(k)]),
                places=3,
            )
        for k in m.F12:
            self.assertAlmostEqual(
                pyo.value(m.F12[k]),
                pyo.value(mex.egb.outputs['F12_{}'.format(k)]),
                places=3,
            )
            self.assertAlmostEqual(
                pyo.value(m.Fo[k]),
                pyo.value(mex.egb.outputs['Fo_{}'.format(k)]),
                places=3,
            )
        """
        self._input_names = ['F1_{}'.format(t) for t in range(1,N)]
        self._input_names.extend(['F2_{}'.format(t) for t in range(1,N)])
        self._input_names.extend(['h1_{}'.format(t) for t in range(0,N)])
        self._input_names.extend(['h2_{}'.format(t) for t in range(0,N)])
        self._output_names = ['F12_{}'.format(t) for t in range(0,N)]
        self._output_names.extend(['Fo_{}'.format(t) for t in range(0,N)])
        """


if __name__ == '__main__':
    t = TestGreyBoxModel()
    t.test_solve()
