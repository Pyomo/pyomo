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

import math
import numpy as np
from scipy.sparse import coo_matrix
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)


class Unconstrained(ExternalGreyBoxModel):
    """
    min (x+2)**2 + (y-2)**2
    """

    def input_names(self):
        return ['x', 'y']

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def has_objective(self):
        return True

    def evaluate_objective(self):
        x = self._input_values[0]
        y = self._input_values[1]
        return (x + 2) ** 2 + (y - 2) ** 2

    def evaluate_grad_objective(self):
        x = self._input_values[0]
        y = self._input_values[1]
        return np.asarray([2 * (x + 2), 2 * (y - 2)], dtype=float)


class Constrained(ExternalGreyBoxModel):
    """
    min x**2 + y**2
    s.t. 0 == y - exp(x)
    """

    def input_names(self):
        return ['x', 'y']

    def set_input_values(self, input_values):
        self._input_values = list(input_values)

    def has_objective(self):
        return True

    def evaluate_objective(self):
        x = self._input_values[0]
        y = self._input_values[1]
        return x**2 + y**2

    def evaluate_grad_objective(self):
        x = self._input_values[0]
        y = self._input_values[1]
        return np.asarray([2 * x, 2 * y], dtype=float)

    def equality_constraint_names(self):
        return ['c1']

    def evaluate_equality_constraints(self):
        x = self._input_values[0]
        y = self._input_values[1]
        return np.asarray([y - math.exp(x)], dtype=float)

    def evaluate_jacobian_equality_constraints(self):
        x = self._input_values[0]
        row = [0, 0]
        col = [0, 1]
        data = [-math.exp(x), 1]
        jac = coo_matrix((data, (row, col)), shape=(1, 2))
        return jac


class ConstrainedWithHessian(Constrained):
    def evaluate_hessian_objective(self):
        row = [0, 1]
        col = [0, 1]
        data = [2, 2]
        hess = coo_matrix((data, (row, col)), shape=(2, 2))
        return hess

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        self._dual = eq_con_multiplier_values[0]

    def evaluate_hessian_equality_constraints(self):
        x = self._input_values[0]
        row = [0]
        col = [0]
        data = [-math.exp(x) * self._dual]
        hess = coo_matrix((data, (row, col)), shape=(2, 2))
        return hess


def solve_unconstrained():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    m.grey_box = ExternalGreyBoxBlock(external_model=Unconstrained())
    m.c = pyo.Constraint(expr=m.z == m.grey_box.inputs['x'] + 1)

    opt = pyo.SolverFactory('cyipopt')
    opt.config.options['hessian_approximation'] = 'limited-memory'
    res = opt.solve(m, tee=True)
    pyo.assert_optimal_termination(res)
    x = m.grey_box.inputs['x'].value
    y = m.grey_box.inputs['y'].value
    assert math.isclose(x, -2)
    assert math.isclose(y, 2)
    return m


def solve_constrained():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    m.grey_box = ExternalGreyBoxBlock(external_model=Constrained())
    m.c2 = pyo.Constraint(expr=m.z == m.grey_box.inputs['x'] + 1)

    opt = pyo.SolverFactory('cyipopt')
    opt.config.options['hessian_approximation'] = 'limited-memory'
    res = opt.solve(m, tee=True)
    pyo.assert_optimal_termination(res)
    x = m.grey_box.inputs['x'].value
    y = m.grey_box.inputs['y'].value
    assert math.isclose(x, -0.4263027509962655)
    assert math.isclose(y, 0.6529186403960969)
    return m


def solve_constrained_with_hessian():
    m = pyo.ConcreteModel()
    m.z = pyo.Var()
    m.grey_box = ExternalGreyBoxBlock(external_model=ConstrainedWithHessian())
    m.c2 = pyo.Constraint(expr=m.z == m.grey_box.inputs['x'] + 1)

    opt = pyo.SolverFactory('cyipopt')
    res = opt.solve(m, tee=True)
    pyo.assert_optimal_termination(res)
    x = m.grey_box.inputs['x'].value
    y = m.grey_box.inputs['y'].value
    assert math.isclose(x, -0.4263027509962655)
    assert math.isclose(y, 0.6529186403960969)
    return m


if __name__ == '__main__':
    m = solve_constrained_with_hessian()
    print(f"x: {m.grey_box.inputs['x'].value}")
    print(f"y: {m.grey_box.inputs['y'].value}")
