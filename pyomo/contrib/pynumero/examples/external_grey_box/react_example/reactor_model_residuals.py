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
"""
This file contains an external grey box model representing a simple
reactor design problem described in the Pyomo book.
It is part of the external_grey_box examples with PyNumero.

Note: In this case, this model can be solved using
standard Pyomo constructs (see the Pyomo book), but
this is included as an example of the external grey
box model interface.
"""


import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel


def create_pyomo_reactor_model():
    # this function is here to show what the "ReactorModel" would
    # look like if it was coded directly in Pyomo
    # create the concrete model
    m = pyo.ConcreteModel()

    # set the data (native python data)
    k1 = 5.0 / 6.0  # min^-1
    k2 = 5.0 / 3.0  # min^-1
    k3 = 1.0 / 6000.0  # m^3/(gmol min)
    # caf = 10000.0    # gmol/m^3

    # create the variables
    m.sv = pyo.Var(initialize=1.0, bounds=(0, None))
    m.caf = pyo.Var(initialize=1.0)
    m.ca = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cb = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cc = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cd = pyo.Var(initialize=1.0, bounds=(0, None))
    m.cb_ratio = pyo.Var(initialize=1.0)

    # create the objective
    m.obj = pyo.Objective(expr=m.cb_ratio, sense=pyo.maximize)

    # create the constraints
    m.ca_bal = pyo.Constraint(
        expr=(0 == m.sv * m.caf - m.sv * m.ca - k1 * m.ca - 2.0 * k3 * m.ca**2.0)
    )

    m.cb_bal = pyo.Constraint(expr=(0 == -m.sv * m.cb + k1 * m.ca - k2 * m.cb))

    m.cc_bal = pyo.Constraint(expr=(0 == -m.sv * m.cc + k2 * m.cb))

    m.cd_bal = pyo.Constraint(expr=(0 == -m.sv * m.cd + k3 * m.ca**2.0))

    m.cb_ratio_con = pyo.Constraint(expr=m.cb / (m.ca + m.cc + m.cd) - m.cb_ratio == 0)
    m.cafcon = pyo.Constraint(expr=m.caf == 10000)

    return m


class ReactorModel(ExternalGreyBoxModel):
    def __init__(self, use_exact_derivatives=True):
        self._use_exact_derivatives = use_exact_derivatives

    def input_names(self):
        return ['sv', 'caf', 'ca', 'cb', 'cc', 'cd']

    def equality_constraint_names(self):
        return ['ca_bal', 'cb_bal', 'cc_bal', 'cd_bal']

    def output_names(self):
        return ['cb_ratio']

    def finalize_block_construction(self, pyomo_block):
        # set lower bounds on the variables
        pyomo_block.inputs['sv'].setlb(0)
        pyomo_block.inputs['ca'].setlb(0)
        pyomo_block.inputs['cb'].setlb(0)
        pyomo_block.inputs['cc'].setlb(0)
        pyomo_block.inputs['cd'].setlb(0)

        # initialize the variables
        pyomo_block.inputs['sv'].value = 1
        pyomo_block.inputs['caf'].value = 1
        pyomo_block.inputs['ca'].value = 1
        pyomo_block.inputs['cb'].value = 1
        pyomo_block.inputs['cc'].value = 1
        pyomo_block.inputs['cd'].value = 1
        pyomo_block.outputs['cb_ratio'].value = 1

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000
        r = np.zeros(4)
        r[0] = sv * caf + (-sv - k1) * ca - 2 * k3 * ca**2
        r[1] = k1 * ca + (-sv - k2) * cb
        r[2] = k2 * cb - sv * cc
        r[3] = k3 * ca**2 - sv * cd
        return r

    def evaluate_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb / (ca + cc + cd)
        return np.asarray([cb_ratio], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000

        if self._use_exact_derivatives:
            row = np.zeros(12)
            col = np.zeros(12)
            data = np.zeros(12)
            row[0], col[0], data[0] = (0, 0, caf - ca)
            row[1], col[1], data[1] = (0, 1, sv)
            row[2], col[2], data[2] = (0, 2, -sv - k1 - 4 * k3 * ca)
            row[3], col[3], data[3] = (1, 0, -cb)
            row[4], col[4], data[4] = (1, 2, k1)
            row[5], col[5], data[5] = (1, 3, -sv - k2)
            row[6], col[6], data[6] = (2, 0, -cc)
            row[7], col[7], data[7] = (2, 3, k2)
            row[8], col[8], data[8] = (2, 4, -sv)
            row[9], col[9], data[9] = (3, 0, -cd)
            row[10], col[10], data[10] = (3, 2, 2 * k3 * ca)
            row[11], col[11], data[11] = (3, 5, -sv)
            ret = coo_matrix((data, (row, col)), shape=(4, 6))
            return ret
        else:
            delta = 1e-8
            u0 = np.copy(self._input_values)
            y0 = self.evaluate_equality_constraints()
            jac = np.empty((4, 6))
            u = np.copy(self._input_values)
            for j in range(len(u)):
                # perturb the variables
                u[j] += delta
                self.set_input_values(u)
                yperturb = self.evaluate_equality_constraints()
                jac_col = (yperturb - y0) / delta
                jac[:, j] = jac_col
                u[j] = u0[j]

            # return us back to our starting state
            self.set_input_values(u0)

            # this needs to be a sparse coo_matrix
            # this approach is inefficient, but clear for an example
            row = []
            col = []
            data = []
            for r in range(4):
                for c in range(6):
                    row.append(r)
                    col.append(c)
                    data.append(jac[r, c])
            ret = coo_matrix((data, (row, col)), shape=(4, 6))
            return ret

    def evaluate_jacobian_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb / (ca + cc + cd)
        row = np.zeros(4)
        col = np.zeros(4)
        data = np.zeros(4)
        row[0], col[0], data[0] = (0, 2, -cb / (ca + cc + cd) ** 2)
        row[1], col[1], data[1] = (0, 3, 1 / (ca + cc + cd))
        row[2], col[2], data[2] = (0, 4, -cb / (ca + cc + cd) ** 2)
        row[3], col[3], data[3] = (0, 5, -cb / (ca + cc + cd) ** 2)
        return coo_matrix((data, (row, col)), shape=(1, 6))


class ReactorModelWithHessian(ReactorModel):
    def __init__(self):
        super(ReactorModelWithHessian, self).__init__(True)
        self._eq_con_mult_values = np.zeros(4)
        self._output_con_mult_values = np.zeros(1)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 4
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 1
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000
        lam = self._eq_con_mult_values

        nnz = 7
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        irow[idx], jcol[idx], data[idx] = (1, 0, lam[0] * 1.0)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (2, 0, lam[0] * (-1.0))
        idx += 1
        irow[idx], jcol[idx], data[idx] = (2, 2, lam[0] * (-4 * k3))
        idx += 1

        irow[idx], jcol[idx], data[idx] = (3, 0, lam[1] * (-1))
        idx += 1

        irow[idx], jcol[idx], data[idx] = (4, 0, lam[2] * (-1))
        idx += 1

        irow[idx], jcol[idx], data[idx] = (2, 2, lam[3] * (2 * k3))
        idx += 1
        irow[idx], jcol[idx], data[idx] = (5, 0, lam[3] * (-1))
        idx += 1

        assert idx == nnz
        hess = coo_matrix((data, (irow, jcol)), shape=(6, 6))
        return hess

    def evaluate_hessian_outputs(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000
        lam = self._output_con_mult_values

        nnz = 9
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)

        h1 = 2 * cb / ((ca + cc + cd) ** 3)
        h2 = -1.0 / ((ca + cc + cd) ** 2)

        idx = 0
        irow[idx], jcol[idx], data[idx] = (2, 2, lam[0] * h1)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (3, 2, lam[0] * h2)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (4, 2, lam[0] * h1)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (4, 3, lam[0] * h2)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (4, 4, lam[0] * h1)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (5, 2, lam[0] * h1)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (5, 3, lam[0] * h2)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (5, 4, lam[0] * h1)
        idx += 1
        irow[idx], jcol[idx], data[idx] = (5, 5, lam[0] * h1)
        idx += 1

        assert idx == nnz
        hess = coo_matrix((data, (irow, jcol)), shape=(6, 6))
        return hess


class ReactorModelNoOutputs(ExternalGreyBoxModel):
    def input_names(self):
        return ['sv', 'caf', 'ca', 'cb', 'cc', 'cd']

    def equality_constraint_names(self):
        return ['ca_bal', 'cb_bal', 'cc_bal', 'cd_bal']

    def output_names(self):
        return []

    def finalize_block_construction(self, pyomo_block):
        # set lower bounds on the variables
        pyomo_block.inputs['sv'].setlb(0)
        pyomo_block.inputs['ca'].setlb(0)
        pyomo_block.inputs['cb'].setlb(0)
        pyomo_block.inputs['cc'].setlb(0)
        pyomo_block.inputs['cd'].setlb(0)

        # initialize the variables
        pyomo_block.inputs['sv'].value = 1
        pyomo_block.inputs['caf'].value = 1
        pyomo_block.inputs['ca'].value = 1
        pyomo_block.inputs['cb'].value = 1
        pyomo_block.inputs['cc'].value = 1
        pyomo_block.inputs['cd'].value = 1

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000
        r = np.zeros(4)
        r[0] = sv * caf + (-sv - k1) * ca - 2 * k3 * ca**2
        r[1] = k1 * ca + (-sv - k2) * cb
        r[2] = k2 * cb - sv * cc
        r[3] = k3 * ca**2 - sv * cd
        return r

    def evaluate_outputs(self):
        raise NotImplementedError()

    def evaluate_jacobian_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000

        row = np.zeros(12)
        col = np.zeros(12)
        data = np.zeros(12)
        row[0], col[0], data[0] = (0, 0, caf - ca)
        row[1], col[1], data[1] = (0, 1, sv)
        row[2], col[2], data[2] = (0, 2, -sv - k1 - 4 * k3 * ca)
        row[3], col[3], data[3] = (1, 0, -cb)
        row[4], col[4], data[4] = (1, 2, k1)
        row[5], col[5], data[5] = (1, 3, -sv - k2)
        row[6], col[6], data[6] = (2, 0, -cc)
        row[7], col[7], data[7] = (2, 3, k2)
        row[8], col[8], data[8] = (2, 4, -sv)
        row[9], col[9], data[9] = (3, 0, -cd)
        row[10], col[10], data[10] = (3, 2, 2 * k3 * ca)
        row[11], col[11], data[11] = (3, 5, -sv)
        ret = coo_matrix((data, (row, col)), shape=(4, 6))
        return ret

    def evaluate_jacobian_outputs(self):
        raise NotImplementedError()


class ReactorModelScaled(ExternalGreyBoxModel):
    def input_names(self):
        return ['sv', 'caf', 'ca', 'cb', 'cc', 'cd']

    def equality_constraint_names(self):
        return ['ca_bal', 'cb_bal', 'cc_bal', 'cd_bal']

    def output_names(self):
        return ['cb_ratio']

    def finalize_block_construction(self, pyomo_block):
        # set lower bounds on the variables
        pyomo_block.inputs['sv'].setlb(0)
        pyomo_block.inputs['ca'].setlb(0)
        pyomo_block.inputs['cb'].setlb(0)
        pyomo_block.inputs['cc'].setlb(0)
        pyomo_block.inputs['cd'].setlb(0)

        # initialize the variables
        pyomo_block.inputs['sv'].value = 1
        pyomo_block.inputs['caf'].value = 1
        pyomo_block.inputs['ca'].value = 1
        pyomo_block.inputs['cb'].value = 1
        pyomo_block.inputs['cc'].value = 1
        pyomo_block.inputs['cd'].value = 1
        pyomo_block.outputs['cb_ratio'].value = 1

        m = pyomo_block.model()
        if not hasattr(m, 'scaling_factor'):
            # add the scaling factor suffix to the model if it is not already declared
            m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.scaling_factor[pyomo_block.inputs['sv']] = 1.1
        m.scaling_factor[pyomo_block.inputs['caf']] = 1.2
        m.scaling_factor[pyomo_block.inputs['ca']] = 1.3
        m.scaling_factor[pyomo_block.inputs['cb']] = 1.4
        m.scaling_factor[pyomo_block.inputs['cc']] = 1.5
        m.scaling_factor[pyomo_block.inputs['cd']] = 1.6
        m.scaling_factor[pyomo_block.outputs['cb_ratio']] = 1.7

    def get_equality_constraint_scaling_factors(self):
        return np.asarray([0.1, 0.2, 0.3, 0.4])

    def get_output_constraint_scaling_factors(self):
        return np.asarray([10])

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000
        r = np.zeros(4)
        r[0] = sv * caf + (-sv - k1) * ca - 2 * k3 * ca**2
        r[1] = k1 * ca + (-sv - k2) * cb
        r[2] = k2 * cb - sv * cc
        r[3] = k3 * ca**2 - sv * cd
        return r

    def evaluate_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb / (ca + cc + cd)
        return np.asarray([cb_ratio], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5 / 6
        k2 = 5 / 3
        k3 = 1 / 6000

        row = np.zeros(12)
        col = np.zeros(12)
        data = np.zeros(12)
        row[0], col[0], data[0] = (0, 0, caf - ca)
        row[1], col[1], data[1] = (0, 1, sv)
        row[2], col[2], data[2] = (0, 2, -sv - k1 - 4 * k3 * ca)
        row[3], col[3], data[3] = (1, 0, -cb)
        row[4], col[4], data[4] = (1, 2, k1)
        row[5], col[5], data[5] = (1, 3, -sv - k2)
        row[6], col[6], data[6] = (2, 0, -cc)
        row[7], col[7], data[7] = (2, 3, k2)
        row[8], col[8], data[8] = (2, 4, -sv)
        row[9], col[9], data[9] = (3, 0, -cd)
        row[10], col[10], data[10] = (3, 2, 2 * k3 * ca)
        row[11], col[11], data[11] = (3, 5, -sv)
        ret = coo_matrix((data, (row, col)), shape=(4, 6))
        return ret

    def evaluate_jacobian_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb / (ca + cc + cd)
        row = np.zeros(4)
        col = np.zeros(4)
        data = np.zeros(4)
        row[0], col[0], data[0] = (0, 2, -cb / (ca + cc + cd) ** 2)
        row[1], col[1], data[1] = (0, 3, 1 / (ca + cc + cd))
        row[2], col[2], data[2] = (0, 4, -cb / (ca + cc + cd) ** 2)
        row[3], col[3], data[3] = (0, 5, -cb / (ca + cc + cd) ** 2)
        return coo_matrix((data, (row, col)), shape=(1, 6))
