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
This file contains a black box model representing a simple
reactor design problem described in the Pyomo book.
It is part of the external_grey_box example with PyNumero.

These functions solve a reactor model using scipy
Note: In this case, this model can be solved using
standard Pyomo constructs (see the Pyomo book), but
this is included as an example of the external grey
box model interface.
"""


import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel


def reactor_outlet_concentrations(sv, caf, k1, k2, k3):
    def _model(x, sv, caf, k1, k2, k3):
        ca, cb, cc, cd = x[0], x[1], x[2], x[3]

        # compute the residuals
        r = np.zeros(4)
        r[0] = sv * caf + (-sv - k1) * ca - 2 * k3 * ca**2
        r[1] = k1 * ca + (-sv - k2) * cb
        r[2] = k2 * cb - sv * cc
        r[3] = k3 * ca**2 - sv * cd

        return r

    concentrations = fsolve(
        lambda x: _model(x, sv, caf, k1, k2, k3), np.ones(4), xtol=1e-8
    )

    # Todo: check solve status
    return concentrations


class ReactorConcentrationsOutputModel(ExternalGreyBoxModel):
    def input_names(self):
        return ['sv', 'caf', 'k1', 'k2', 'k3']

    def output_names(self):
        return ['ca', 'cb', 'cc', 'cd']

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def finalize_block_construction(self, pyomo_block):
        # set lower bounds on the inputs and outputs
        pyomo_block.inputs['sv'].setlb(0)
        pyomo_block.outputs['ca'].setlb(0)
        pyomo_block.outputs['cb'].setlb(0)
        pyomo_block.outputs['cc'].setlb(0)
        pyomo_block.outputs['cd'].setlb(0)

        # initialize the variables
        pyomo_block.inputs['sv'].value = 5
        pyomo_block.inputs['caf'].value = 10000
        pyomo_block.inputs['k1'].value = 5 / 6
        pyomo_block.inputs['k2'].value = 5 / 3
        pyomo_block.inputs['k3'].value = 1 / 6000
        pyomo_block.outputs['ca'].value = 1
        pyomo_block.outputs['cb'].value = 1
        pyomo_block.outputs['cc'].value = 1
        pyomo_block.outputs['cd'].value = 1

    def evaluate_outputs(self):
        sv = self._input_values[0]
        caf = self._input_values[1]
        k1 = self._input_values[2]
        k2 = self._input_values[3]
        k3 = self._input_values[4]
        ret = reactor_outlet_concentrations(sv, caf, k1, k2, k3)
        return np.asarray(ret, dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        # here, we compute the derivatives using finite difference
        # however, this would be better with analytical derivatives
        delta = 1e-6
        u0 = np.copy(self._input_values)
        y0 = self.evaluate_outputs()
        jac = np.empty((4, 5))
        u = np.copy(self._input_values)
        for j in range(len(u)):
            # perturb the variables
            u[j] += delta
            self.set_input_values(u)
            yperturb = self.evaluate_outputs()
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
            for c in range(5):
                row.append(r)
                col.append(c)
                data.append(jac[r, c])

        return coo_matrix((data, (row, col)), shape=(4, 5))


if __name__ == '__main__':
    sv = 1.34
    caf = 10000
    k1 = 5 / 6
    k2 = 5 / 3
    k3 = 1 / 6000
    concentrations = reactor_outlet_concentrations(sv, caf, k1, k2, k3)
    print(concentrations)
