# This file contains an external grey box model representing a simple
# reactor design problem described in the Pyomo book.
# It is part of the external_grey_box examples with PyNumero.

# Note: In this case, this model can be solved using
# standard Pyomo constructs (see the Pyomo book), but
# this is included as an example of the external grey
# box model interface.
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

class ReactorModel(ExternalGreyBoxModel):
    def __init__(self, use_exact_derivatives=True):
        self._use_exact_derivatives = use_exact_derivatives
        
    def input_names(self):
        return ['sv', 'caf', 'ca', 'cb', 'cc', 'cd']

    def equality_constraint_names(self):
        return ['ca_bal', 'cb_bal', 'cc_bal', 'cd_bal']
    
    def output_names(self):
        return ['cb_ratio']

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5/6
        k2 = 5/3
        k3 = 1/6000
        r = np.zeros(4)
        r[0] = sv*caf + (-sv-k1)*ca - 2*k3*ca**2
        r[1] = k1*ca + (-sv-k2)*cb
        r[2] = k2*cb - sv*cc
        r[3] = k3*ca**2 - sv*cd
        return r
    
    def evaluate_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb/(ca+cc+cd)
        return np.asarray([cb_ratio], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5/6
        k2 = 5/3
        k3 = 1/6000

        if self._use_exact_derivatives:
            row = np.zeros(12)
            col = np.zeros(12)
            data = np.zeros(12)
            row[0], col[0], data[0] = (0, 0, caf-ca)
            row[1], col[1], data[1] = (0, 1, sv)
            row[2], col[2], data[2] = (0, 2, -sv-k1-4*k3*ca)
            row[3], col[3], data[3] = (1, 0, -cb)
            row[4], col[4], data[4] = (1, 2, k1)
            row[5], col[5], data[5] = (1, 3, -sv-k2)
            row[6], col[6], data[6] = (2, 0, -cc)
            row[7], col[7], data[7] = (2, 3, k2)
            row[8], col[8], data[8] = (2, 4, -sv)
            row[9], col[9], data[9] = (3, 0, -cd)
            row[10], col[10], data[10] = (3, 2, 2*k3*ca)
            row[11], col[11], data[11] = (3, 5, -sv)
            ret = coo_matrix((data, (row, col)), shape=(4,6))
            return ret
        else:
            delta = 1e-8
            u0 = np.copy(self._input_values)
            y0 = self.evaluate_equality_constraints()
            jac = np.empty((4,6))
            u = np.copy(self._input_values)
            for j in range(len(u)):
                # perturb the variables
                u[j] += delta
                self.set_input_values(u)
                yperturb = self.evaluate_equality_constraints()
                jac_col = (yperturb - y0)/delta
                jac[:,j] = jac_col
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
                    data.append(jac[r,c])
            ret = coo_matrix((data, (row, col)), shape=(4,6))
            return ret

    def evaluate_jacobian_outputs(self):
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        cb_ratio = cb/(ca+cc+cd)
        row = np.zeros(4)
        col = np.zeros(4)
        data = np.zeros(4)
        row[0], col[0], data[0] = (0, 2, -cb/(ca+cc+cd))
        row[1], col[1], data[1] = (0, 3, 1/(ca+cc+cd))
        row[2], col[2], data[2] = (0, 4, -cb/(ca+cc+cd))
        row[3], col[3], data[3] = (0, 5, -cb/(ca+cc+cd))
        return coo_matrix((data, (row, col)), shape=(1,6))


class ReactorModelNoOutputs(ExternalGreyBoxModel):
    def input_names(self):
        return ['sv', 'caf', 'ca', 'cb', 'cc', 'cd']

    def equality_constraint_names(self):
        return ['ca_bal', 'cb_bal', 'cc_bal', 'cd_bal']
    
    def output_names(self):
        return []

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        ca = self._input_values[2]
        cb = self._input_values[3]
        cc = self._input_values[4]
        cd = self._input_values[5]
        k1 = 5/6
        k2 = 5/3
        k3 = 1/6000
        r = np.zeros(4)
        r[0] = sv*caf + (-sv-k1)*ca - 2*k3*ca**2
        r[1] = k1*ca + (-sv-k2)*cb
        r[2] = k2*cb - sv*cc
        r[3] = k3*ca**2 - sv*cd
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
        k1 = 5/6
        k2 = 5/3
        k3 = 1/6000

        row = np.zeros(12)
        col = np.zeros(12)
        data = np.zeros(12)
        row[0], col[0], data[0] = (0, 0, caf-ca)
        row[1], col[1], data[1] = (0, 1, sv)
        row[2], col[2], data[2] = (0, 2, -sv-k1-4*k3*ca)
        row[3], col[3], data[3] = (1, 0, -cb)
        row[4], col[4], data[4] = (1, 2, k1)
        row[5], col[5], data[5] = (1, 3, -sv-k2)
        row[6], col[6], data[6] = (2, 0, -cc)
        row[7], col[7], data[7] = (2, 3, k2)
        row[8], col[8], data[8] = (2, 4, -sv)
        row[9], col[9], data[9] = (3, 0, -cd)
        row[10], col[10], data[10] = (3, 2, 2*k3*ca)
        row[11], col[11], data[11] = (3, 5, -sv)
        ret = coo_matrix((data, (row, col)), shape=(4,6))
        return ret

    def evaluate_jacobian_outputs(self):
        raise NotImplementedError()

    
