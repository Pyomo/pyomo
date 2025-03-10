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

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.dependencies.scipy import sparse as spa

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock

# set of external models for testing
# basic model is a simple pipe sequence with nonlinear pressure drop
# Pin -> P1 -> P2 -> P3 -> Pout
#
# We will assume that we have an external model to compute
# the pressure drop in this sequence of pipes, where the dP
# is given by c*F^2
#
# There are several ways to format this.
# Model 1: Use the "external model" to compute the output pressure
#   no equalities, 1 output
#   u = [Pin, c, F]
#   o = [Pout]
#   h_eq(u) = {empty}
#   h_o(u) = [Pin - 4*c*F^2]
#
# Model 2: Same as model 1, but treat Pout as an input to be converged by the optimizer
#   1 equality, no outputs
#   u = [Pin, c, F, Pout]
#   o = {empty}
#   h_eq(u) = [Pout - (Pin - 4*c*F^2]
#   h_o(u) = {empty}
#
# Model 3: Use the "external model" to compute the output pressure and the pressure
#          at node 2 (e.g., maybe we have a measurement there we want to match)
#   no equalities, 2 outputs
#   u = [Pin, c, F]
#   o = [P2, Pout]
#   h_eq(u) = {empty}
#   h_o(u) = [Pin - 2*c*F^2]
#            [Pin - 4*c*F^2]
#
# Model 4: Same as model 2, but treat P2, and Pout as an input to be converged by the optimizer
#   2 equality, no outputs
#   u = [Pin, c, F, P2, Pout]
#   o = {empty}
#   h_eq(u) = [P2 - (Pin - 2*c*F^2]
#             [Pout - (P2 - 2*c*F^2]
#   h_o(u) = {empty}


# Model 4: Same as model 2, but treat P2 as an input to be converged by the solver
#   u = [Pin, c, F, P2]
#   o = [Pout]
#   h_eq(u) = P2 - (Pin-2*c*F^2)]
#   h_o(u) = [Pin - 4*c*F^2] (or could also be [P2 - 2*c*F^2])
#
# Model 5: treat all "internal" variables as "inputs", equality and output equations
#   u = [Pin, c, F, P1, P2, P3]
#   o = [Pout]
#    h_eq(u) = [
#               P1 - (Pin - c*F^2);
#               P2 - (P1 - c*F^2);
#               P3 - (P2 - c*F^2);
#              ]
#   h_o(u) = [P3 - c*F^2] (or could also be [Pin - 4*c*F^2] or [P1 - 3*c*F^2] or [P2 - 2*c*F^2])
#
# Model 6: treat all variables as "inputs", equality only, and no output equations
#   u = [Pin, c, F, P1, P2, P3, Pout]
#   o = {empty}
#   h_eq(u) = [
#               P1 - (Pin - c*F^2);
#               P2 - (P1 - c*F^2);
#               P3 - (P2 - c*F^2);
#               Pout = (P3 - c*F^2);
#              ]
#   h_o(u) = {empty}
#
class PressureDropSingleOutput(ExternalGreyBoxModel):
    def __init__(self):
        self._input_names = ['Pin', 'c', 'F']
        self._input_values = np.zeros(3, dtype=np.float64)
        self._output_names = ['Pout']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return []

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 3
        np.copyto(self._input_values, input_values)

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = Pin - 4 * c * F**2
        return np.asarray([Pout], dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([1, -4 * F**2, -4 * c * 2 * F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 3))
        return jac


class PressureDropSingleOutputWithHessian(PressureDropSingleOutput):
    def __init__(self):
        super(PressureDropSingleOutputWithHessian, self).__init__()
        self._output_con_mult_values = np.zeros(1, dtype=np.float64)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        data = self._output_con_mult_values[0] * np.asarray(
            [-8 * F, -8 * c], dtype=np.float64
        )
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(3, 3))
        return hess


class PressureDropSingleEquality(ExternalGreyBoxModel):
    #   u = [Pin, c, F, Pout]
    #   o = {empty}
    #   h_eq(u) = [Pout - (Pin - 4*c*F^2]
    #   h_o(u) = {empty}
    def __init__(self):
        self._input_names = ['Pin', 'c', 'F', 'Pout']
        self._input_values = np.zeros(4, dtype=np.float64)
        self._equality_constraint_names = ['pdrop']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def set_input_values(self, input_values):
        assert len(input_values) == 4
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = self._input_values[3]
        return np.asarray([Pout - (Pin - 4 * c * F**2)], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3], dtype=np.int64)
        nonzeros = np.asarray([-1, 4 * F**2, 4 * 2 * c * F, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 4))
        return jac


class PressureDropSingleEqualityWithHessian(PressureDropSingleEquality):
    #   u = [Pin, c, F, Pout]
    #   o = {empty}
    #   h_eq(u) = [Pout - (Pin - 4*c*F^2]
    #   h_o(u) = {empty}
    def __init__(self):
        super(PressureDropSingleEqualityWithHessian, self).__init__()
        self._eq_con_mult_values = np.zeros(1, dtype=np.float64)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 1
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = self._eq_con_mult_values[0] * np.asarray(
            [8 * F, 8 * c], dtype=np.float64
        )
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(4, 4))
        return hess


class PressureDropTwoOutputs(ExternalGreyBoxModel):
    #   u = [Pin, c, F]
    #   o = [P2, Pout]
    #   h_eq(u) = {empty}
    #   h_o(u) = [Pin - 2*c*F^2]
    #            [Pin - 4*c*F^2]
    def __init__(self):
        self._input_names = ['Pin', 'c', 'F']
        self._input_values = np.zeros(3, dtype=np.float64)
        self._output_names = ['P2', 'Pout']

    def input_names(self):
        return self._input_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 3
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P2 = Pin - 2 * c * F**2
        Pout = Pin - 4 * c * F**2
        return np.asarray([P2, Pout], dtype=np.float64)

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [1, -2 * F**2, -2 * c * 2 * F, 1, -4 * F**2, -4 * c * 2 * F],
            dtype=np.float64,
        )
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 3))
        return jac


class PressureDropTwoOutputsWithHessian(PressureDropTwoOutputs):
    #   u = [Pin, c, F]
    #   o = [P2, Pout]
    #   h_eq(u) = {empty}
    #   h_o(u) = [Pin - 2*c*F^2]
    #            [Pin - 4*c*F^2]
    def __init__(self):
        super(PressureDropTwoOutputsWithHessian, self).__init__()
        self._output_con_mult_values = np.zeros(2, dtype=np.float64)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._output_con_mult_values[0]
        y2 = self._output_con_mult_values[1]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [y1 * (-4 * F) + y2 * (-8 * F), y1 * (-4 * c) + y2 * (-8 * c)],
            dtype=np.float64,
        )
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(3, 3))
        return hess


class PressureDropTwoEqualities(ExternalGreyBoxModel):
    #   u = [Pin, c, F, P2, Pout]
    #   o = {empty}
    #   h_eq(u) = [P2 - (Pin - 2*c*F^2]
    #             [Pout - (P2 - 2*c*F^2]
    #   h_o(u) = {empty}
    def __init__(self):
        self._input_names = ['Pin', 'c', 'F', 'P2', 'Pout']
        self._input_values = np.zeros(5, dtype=np.float64)
        self._equality_constraint_names = ['pdrop2', 'pdropout']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def set_input_values(self, input_values):
        assert len(input_values) == 5
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P2 = self._input_values[3]
        Pout = self._input_values[4]
        return np.asarray(
            [P2 - (Pin - 2 * c * F**2), Pout - (P2 - 2 * c * F**2)], dtype=np.float64
        )

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
        nonzeros = np.asarray(
            [-1, 2 * F**2, 2 * 2 * c * F, 1, 2 * F**2, 2 * 2 * c * F, -1, 1],
            dtype=np.float64,
        )
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 5))
        return jac


class PressureDropTwoEqualitiesWithHessian(PressureDropTwoEqualities):
    #   u = [Pin, c, F, P2, Pout]
    #   o = {empty}
    #   h_eq(u) = [P2 - (Pin - 2*c*F^2]
    #             [Pout - (P2 - 2*c*F^2]
    #   h_o(u) = {empty}
    def __init__(self):
        super(PressureDropTwoEqualitiesWithHessian, self).__init__()
        self._eq_con_mult_values = np.zeros(2, dtype=np.float64)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._eq_con_mult_values[0]
        y2 = self._eq_con_mult_values[1]

        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [y1 * (4 * F) + y2 * (4 * F), y1 * (4 * c) + y2 * (4 * c)], dtype=np.float64
        )
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(5, 5))
        return hess


class PressureDropTwoEqualitiesTwoOutputs(ExternalGreyBoxModel):
    #   u = [Pin, c, F, P1, P3]
    #   o = {P2, Pout}
    #   h_eq(u) = [P1 - (Pin - c*F^2]
    #             [P3 - (P1 - 2*c*F^2]
    #   h_o(u) = [P1 - c*F^2]
    #            [Pin - 4*c*F^2]
    def __init__(self):
        self._input_names = ['Pin', 'c', 'F', 'P1', 'P3']
        self._input_values = np.zeros(5, dtype=np.float64)
        self._equality_constraint_names = ['pdrop1', 'pdrop3']
        self._output_names = ['P2', 'Pout']

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 5
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P1 = self._input_values[3]
        P3 = self._input_values[4]
        return np.asarray(
            [P1 - (Pin - c * F**2), P3 - (P1 - 2 * c * F**2)], dtype=np.float64
        )

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P1 = self._input_values[3]
        return np.asarray([P1 - c * F**2, Pin - 4 * c * F**2], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
        nonzeros = np.asarray(
            [-1, F**2, 2 * c * F, 1, 2 * F**2, 4 * c * F, -1, 1], dtype=np.float64
        )
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 5))
        return jac

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [-(F**2), -c * 2 * F, 1, 1, -4 * F**2, -4 * c * 2 * F], dtype=np.float64
        )
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2, 5))
        return jac


class PressureDropTwoEqualitiesTwoOutputsWithHessian(
    PressureDropTwoEqualitiesTwoOutputs
):
    #   u = [Pin, c, F, P1, P3]
    #   o = {P2, Pout}
    #   h_eq(u) = [P1 - (Pin - c*F^2]
    #             [P3 - (P1 - 2*c*F^2]
    #   h_o(u) = [P1 - c*F^2]
    #            [Pin - 4*c*F^2]
    def __init__(self):
        super(PressureDropTwoEqualitiesTwoOutputsWithHessian, self).__init__()
        self._eq_con_mult_values = np.zeros(2, dtype=np.float64)
        self._output_con_mult_values = np.zeros(2, dtype=np.float64)

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 2
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_hessian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._eq_con_mult_values[0]
        y2 = self._eq_con_mult_values[1]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [y1 * (2 * F) + y2 * (4 * F), y1 * (2 * c) + y2 * (4 * c)], dtype=np.float64
        )
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(5, 5))
        return hess

    def evaluate_hessian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        y1 = self._output_con_mult_values[0]
        y2 = self._output_con_mult_values[1]
        irow = np.asarray([2, 2], dtype=np.int64)
        jcol = np.asarray([1, 2], dtype=np.int64)
        nonzeros = np.asarray(
            [y1 * (-2 * F) + y2 * (-8 * F), y1 * (-2 * c) + y2 * (-8 * c)],
            dtype=np.float64,
        )
        hess = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(5, 5))
        return hess


class PressureDropTwoEqualitiesTwoOutputsScaleBoth(PressureDropTwoEqualitiesTwoOutputs):
    def get_equality_constraint_scaling_factors(self):
        return np.asarray([3.1, 3.2], dtype=np.float64)

    def get_output_constraint_scaling_factors(self):
        return np.asarray([4.1, 4.2])


class PressureDropTwoEqualitiesTwoOutputsScaleEqualities(
    PressureDropTwoEqualitiesTwoOutputs
):
    def get_equality_constraint_scaling_factors(self):
        return np.asarray([3.1, 3.2], dtype=np.float64)


class PressureDropTwoEqualitiesTwoOutputsScaleOutputs(
    PressureDropTwoEqualitiesTwoOutputs
):
    def get_output_constraint_scaling_factors(self):
        return np.asarray([4.1, 4.2])


class OneOutput(ExternalGreyBoxModel):
    def __init__(self):
        self._input_names = ['u']
        self._output_names = ['o']
        self._u = None
        self._output_mult = None

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return []

    def output_names(self):
        return self._output_names

    def finalize_block_construction(self, pyomo_block):
        pyomo_block.inputs['u'].setlb(4)
        pyomo_block.inputs['u'].setub(10)
        pyomo_block.inputs['u'].set_value(1.0)
        pyomo_block.outputs['o'].set_value(1.0)

    def set_input_values(self, input_values):
        assert len(input_values) == 1
        self._u = input_values[0]

    def evaluate_outputs(self):
        return np.asarray([5 * self._u])

    def evaluate_jacobian_outputs(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        nonzeros = np.asarray([5.0], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 1))
        return jac

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 1
        self._output_mult = output_con_multiplier_values[0]

    def evaluate_hessian_outputs(self):
        irow = np.asarray([], dtype=np.int64)
        jcol = np.asarray([], dtype=np.int64)
        data = np.asarray([], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(1, 1))
        return hess


class OneOutputOneEquality(ExternalGreyBoxModel):
    def __init__(self):
        self._input_names = ['u']
        self._equality_constraint_names = ['u2_con']
        self._output_names = ['o']
        self._u = None
        self._output_mult = None
        self._equality_mult = None

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def finalize_block_construction(self, pyomo_block):
        pyomo_block.inputs['u'].set_value(1.0)
        pyomo_block.outputs['o'].set_value(1.0)

    def set_input_values(self, input_values):
        assert len(input_values) == 1
        self._u = input_values[0]

    def evaluate_equality_constraints(self):
        return np.asarray([self._u**2 - 1])

    def evaluate_outputs(self):
        return np.asarray([5 * self._u])

    def evaluate_jacobian_equality_constraints(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        nonzeros = np.asarray([2 * self._u], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 1))
        return jac

    def evaluate_jacobian_outputs(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        nonzeros = np.asarray([5.0], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1, 1))
        return jac

    def set_equality_constraint_multipliers(self, equality_con_multiplier_values):
        assert len(equality_con_multiplier_values) == 1
        self._equality_mult = equality_con_multiplier_values[0]

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 1
        self._output_mult = output_con_multiplier_values[0]

    def evaluate_hessian_equality_constraints(self):
        irow = np.asarray([0], dtype=np.int64)
        jcol = np.asarray([0], dtype=np.int64)
        data = np.asarray([self._equality_mult * 2.0], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(1, 1))
        return hess

    def evaluate_hessian_outputs(self):
        irow = np.asarray([], dtype=np.int64)
        jcol = np.asarray([], dtype=np.int64)
        data = np.asarray([], dtype=np.float64)
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(1, 1))
        return hess
