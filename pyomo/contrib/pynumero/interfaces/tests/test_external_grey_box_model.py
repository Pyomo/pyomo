#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import pyutilib.th as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_sparse as spa, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

try:
    import ipopt
except ImportError:
    raise unittest.SkipTest("Pynumero needs cyipopt to run CyIpoptSolver tests")

from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock, _ExternalGreyBoxModelHelper
from ..pyomo_nlp import PyomoGreyBoxNLP

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptSolver, CyIpoptNLP
)

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

    def evaluate_equality_constraints(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = Pin - 4*c*F**2
        return np.asarray([Pout], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([1, -4*F**2, -4*c*2*F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1,3))
        return jac

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

    def output_names(self):
        return []

    def set_input_values(self, input_values):
        assert len(input_values) == 4
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        Pout = self._input_values[3]
        return np.asarray([Pout - (Pin - 4*c*F**2)], dtype=np.float64)

    def evaluate_outputs(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3], dtype=np.int64)
        nonzeros = np.asarray([-1, 4*F**2, 4*2*c*F, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(1,4))
        return jac

    def evaluate_jacobian_outputs(self):
        raise NotImplementedError('This method should not be called for this model.')

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

    def equality_constraint_names(self):
        return []

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
        P2 = Pin - 2*c*F**2
        Pout = Pin - 4*c*F**2
        return np.asarray([P2, Pout], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([1, -2*F**2, -2*c*2*F, 1, -4*F**2, -4*c*2*F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2,3))
        return jac

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

    def output_names(self):
        return []

    def set_input_values(self, input_values):
        assert len(input_values) == 5
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P2 = self._input_values[3]
        Pout = self._input_values[4]
        return np.asarray([P2 - (Pin - 2*c*F**2), Pout - (P2 - 2*c*F**2)], dtype=np.float64)

    def evaluate_outputs(self):
        raise NotImplementedError('This method should not be called for this model.')

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
        nonzeros = np.asarray([-1, 2*F**2, 2*2*c*F, 1, 2*F**2, 2*2*c*F, -1, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2,5))
        return jac

    def evaluate_jacobian_outputs(self):
        raise NotImplementedError('This method should not be called for this model.')

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
        return np.asarray([P1 - (Pin - c*F**2), P3 - (P1 - 2*c*F**2)], dtype=np.float64)

    def evaluate_outputs(self):
        Pin = self._input_values[0]
        c = self._input_values[1]
        F = self._input_values[2]
        P1 = self._input_values[3]
        return np.asarray([P1 - c*F**2, Pin - 4*c*F**2], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([0, 1, 2, 3, 1, 2, 3, 4], dtype=np.int64)
        nonzeros = np.asarray([-1, F**2, 2*c*F, 1, 2*F**2, 4*c*F, -1, 1], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2,5))
        return jac

    def evaluate_jacobian_outputs(self):
        c = self._input_values[1]
        F = self._input_values[2]
        irow = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        jcol = np.asarray([1, 2, 3, 0, 1, 2], dtype=np.int64)
        nonzeros = np.asarray([-F**2, -c*2*F, 1, 1, -4*F**2, -4*c*2*F], dtype=np.float64)
        jac = spa.coo_matrix((nonzeros, (irow, jcol)), shape=(2,5))
        return jac

class TestExternalGreyBoxModel(unittest.TestCase):

    def test_pressure_drop_single_output(self):
        egbm = PressureDropSingleOutput()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, [])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0,0,0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0,1,2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1,-36,-48], dtype=np.float64)))

    def test_pressure_drop_single_equality(self):
        egbm = PressureDropSingleEquality()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, [])

        egbm.set_input_values(np.asarray([100, 2, 3, 50], dtype=np.float64))

        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([22], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0,0,0,0], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0,1,2,3], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 36, 48, 1], dtype=np.float64)))

    def test_pressure_drop_two_outputs(self):
        egbm = PressureDropTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual([], eq_con_names)
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3], dtype=np.float64))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_equality_constraints()

        #   u = [Pin, c, F]
        #   o = [P2, Pout]
        #   h_eq(u) = {empty}
        #   h_o(u) = [Pin - 2*c*F^2]
        #            [Pin - 4*c*F^2]
            
        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([64, 28], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_equality_constraints()

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0,0,0,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([0,1,2,0,1,2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([1, -18, -24, 1,-36,-48], dtype=np.float64)))

    def test_pressure_drop_two_equalities(self):
        egbm = PressureDropTwoEqualities()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P2', 'Pout'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop2', 'pdropout'])
        output_names = egbm.output_names()
        self.assertEqual([], output_names)

        egbm.set_input_values(np.asarray([100, 2, 3, 20, 50], dtype=np.float64))

        #   u = [Pin, c, F, P2, Pout]
        #   o = {empty}
        #   h_eq(u) = [P2 - (Pin - 2*c*F^2]
        #             [Pout - (P2 - 2*c*F^2]
        #   h_o(u) = {empty}
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-44, 66], dtype=np.float64)))

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_outputs()

        with self.assertRaises(NotImplementedError):
            tmp = egbm.evaluate_jacobian_outputs()

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0,0,0,0,1,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0,1,2,3,1,2,3,4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 18, 24, 1, 18, 24, -1, 1], dtype=np.float64)))

    def test_pressure_drop_two_equalities_two_outputs(self):
        #   u = [Pin, c, F, P1, P3]
        #   o = {P2, Pout}
        #   h_eq(u) = [P1 - (Pin - c*F^2]
        #             [P3 - (Pin - 2*c*F^2]
        #   h_o(u) = [P1 - c*F^2]
        #            [Pin - 4*c*F^2]
        egbm = PressureDropTwoEqualitiesTwoOutputs()
        input_names = egbm.input_names()
        self.assertEqual(input_names, ['Pin', 'c', 'F', 'P1', 'P3'])
        eq_con_names = egbm.equality_constraint_names()
        self.assertEqual(eq_con_names, ['pdrop1', 'pdrop3'])
        output_names = egbm.output_names()
        self.assertEqual(output_names, ['P2', 'Pout'])

        egbm.set_input_values(np.asarray([100, 2, 3, 80, 70], dtype=np.float64))
        eq = egbm.evaluate_equality_constraints()
        self.assertTrue(np.array_equal(eq, np.asarray([-2, 26], dtype=np.float64)))

        o = egbm.evaluate_outputs()
        self.assertTrue(np.array_equal(o, np.asarray([62, 28], dtype=np.float64)))

        jac_eq = egbm.evaluate_jacobian_equality_constraints()
        self.assertTrue(np.array_equal(jac_eq.row, np.asarray([0,0,0,0,1,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.col, np.asarray([0,1,2,3,1,2,3,4], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_eq.data, np.asarray([-1, 9, 12, 1, 18, 24, -1, 1], dtype=np.float64)))

        jac_o = egbm.evaluate_jacobian_outputs()
        self.assertTrue(np.array_equal(jac_o.row, np.asarray([0,0,0,1,1,1], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.col, np.asarray([1,2,3,0,1,2], dtype=np.int64)))
        self.assertTrue(np.array_equal(jac_o.data, np.asarray([-9, -12, 1, 1, -36, -48], dtype=np.float64)))



# TODO: make this work even if there is only external and no variables anywhere in pyomo part
class TestPyomoGreyBoxNLP(unittest.TestCase):
    @unittest.skip("It looks like ASL exits when there are no variables")
    def test_error_no_variables(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropSingleOutput())
        m.obj = pyo.Objective(expr=1)
        with self.assertRaises(ValueError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

    def test_error_fixed_inputs_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropSingleOutput())
        m.egb.inputs['Pin'].fix(100)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        with self.assertRaises(NotImplementedError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoOutputs())
        m.egb.outputs['P2'].fix(50)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        with self.assertRaises(NotImplementedError):
            pyomo_nlp = PyomoGreyBoxNLP(m)

    def test_pressure_drop_single_output(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropSingleOutput())
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(0)
        m.egb.outputs['Pout'].setub(100)
        #m.dummy = pyo.Constraint(expr=sum(m.egb.inputs[i] for i in m.egb.inputs) + sum(m.egb.outputs[i] for i in m.egb.outputs) <= 1e6)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(4, pyomo_nlp.n_primals())
        self.assertEqual(1, pyomo_nlp.n_constraints())
        self.assertEqual(4, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.outputs[Pout]']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(4, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -36, -48, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_single_equality(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropSingleEquality())
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['Pout'].value = 50
        m.egb.inputs['Pout'].setlb(0)
        m.egb.inputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(4, pyomo_nlp.n_primals())
        self.assertEqual(1, pyomo_nlp.n_constraints())
        self.assertEqual(4, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[Pout]']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(4, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(1)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 36, 48, 1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoOutputs())
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.outputs['P2'].value = 80
        m.egb.outputs['P2'].setlb(10)
        m.egb.outputs['P2'].setub(90)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(0)
        m.egb.outputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(5, pyomo_nlp.n_primals())
        self.assertEqual(2, pyomo_nlp.n_constraints())
        self.assertEqual(8, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.outputs[P2]', 'egb.outputs[Pout]']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.P2_con', 'egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1, 1], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(5, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1, 1], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1, 1], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-16, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[1, -18, -24, -1, 0], [1, -36, -48, 0, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoEqualities())
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P2'].value = 80
        m.egb.inputs['P2'].setlb(10)
        m.egb.inputs['P2'].setub(90)
        m.egb.inputs['Pout'].value = 50
        m.egb.inputs['Pout'].setlb(0)
        m.egb.inputs['Pout'].setub(100)
        m.obj = pyo.Objective(expr=(m.egb.inputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(5, pyomo_nlp.n_primals())
        self.assertEqual(2, pyomo_nlp.n_constraints())
        self.assertEqual(8, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[P2]', 'egb.inputs[Pout]']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop2', 'egb.pdropout']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 100], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1, 1], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(5, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(2, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1, 1], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1, 1], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([16, 6], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(2)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1, 18, 24, 1, 0], [0, 18, 24, -1, 1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_pressure_drop_two_equalities_two_outputs(self):
        m = pyo.ConcreteModel()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoEqualitiesTwoOutputs())
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(7, pyomo_nlp.n_primals())
        self.assertEqual(4, pyomo_nlp.n_constraints())
        self.assertEqual(16, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]',
                              'egb.inputs[P1]', 'egb.inputs[P3]',
                              'egb.outputs[P2]', 'egb.outputs[Pout]']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.P2_con', 'egb.Pout_con']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 80, 85, 70], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 70, 75, 50], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1, 1, 1, 1], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(7, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(4, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(0, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5,6,7], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10, 11, 12], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10, 11, 12], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1, 1, 1, 1], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1, 1, 1, 1], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 0, 0, 60], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-2, 26, -13, -22], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(4)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1,   9,  12,  1, 0,  0,  0],
                                   [ 0,  18,  24, -1, 1,  0,  0],
                                   [ 0,  -9, -12,  1, 0, -1,  0],
                                   [ 1, -36, -48,  0, 0,  0, -1]])
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_external_additional_constraints_vars(self):
        m = pyo.ConcreteModel()
        m.hin = pyo.Var(bounds=(0,None), initialize=10)
        m.hout = pyo.Var(bounds=(0,None))
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoEqualitiesTwoOutputs())
        m.incon = pyo.Constraint(expr= 0 <= m.egb.inputs['Pin'] - 10*m.hin)
        m.outcon = pyo.Constraint(expr= 0 == m.egb.outputs['Pout'] - 10*m.hout)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        self.assertEqual(9, pyomo_nlp.n_primals())
        self.assertEqual(6, pyomo_nlp.n_constraints())
        self.assertEqual(20, pyomo_nlp.nnz_jacobian())
        with self.assertRaises(NotImplementedError):
            tmp = pyomo_nlp.nnz_hessian_lag()

        comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]',
                              'egb.inputs[P1]', 'egb.inputs[P3]',
                              'egb.outputs[P2]', 'egb.outputs[Pout]',
                              'hin', 'hout']
        x_order = pyomo_nlp.variable_names()
        comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.P2_con', 'egb.Pout_con', 'incon', 'outcon']
        c_order = pyomo_nlp.constraint_names()

        xlb = pyomo_nlp.primals_lb()
        comparison_xlb = np.asarray([50, 1, 1, 10, 20, 15, 30, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, xlb, x_order, comparison_xlb, comparison_x_order)
        xub = pyomo_nlp.primals_ub()
        comparison_xub = np.asarray([150, 5, 5, 90, 80, 85, 70, np.inf, np.inf], dtype=np.float64)
        check_vectors_specific_order(self, xub, x_order, comparison_xub, comparison_x_order)
        clb = pyomo_nlp.constraints_lb()
        comparison_clb = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, clb, c_order, comparison_clb, comparison_c_order)
        cub = pyomo_nlp.constraints_ub()
        comparison_cub = np.asarray([0, 0, 0, 0, np.inf, 0], dtype=np.float64)
        check_vectors_specific_order(self, cub, c_order, comparison_cub, comparison_c_order)

        xinit = pyomo_nlp.init_primals()
        comparison_xinit = np.asarray([100, 2, 3, 80, 70, 75, 50, 10, 0], dtype=np.float64)
        check_vectors_specific_order(self, xinit, x_order, comparison_xinit, comparison_x_order)
        duals_init = pyomo_nlp.init_duals()
        comparison_duals_init = np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, duals_init, c_order, comparison_duals_init, comparison_c_order)

        self.assertEqual(9, len(pyomo_nlp.create_new_vector('primals')))
        self.assertEqual(6, len(pyomo_nlp.create_new_vector('constraints')))
        self.assertEqual(6, len(pyomo_nlp.create_new_vector('duals')))
        self.assertEqual(5, len(pyomo_nlp.create_new_vector('eq_constraints')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('ineq_constraints')))
        self.assertEqual(5, len(pyomo_nlp.create_new_vector('duals_eq')))
        self.assertEqual(1, len(pyomo_nlp.create_new_vector('duals_ineq')))

        pyomo_nlp.set_primals(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64))
        x = pyomo_nlp.get_primals()
        self.assertTrue(np.array_equal(x, np.asarray([1,2,3,4,5,6,7,8,9], dtype=np.float64)))
        pyomo_nlp.set_primals(pyomo_nlp.init_primals())

        pyomo_nlp.set_duals(np.asarray([42, 10, 11, 12, 13, 14], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([42, 10, 11, 12, 13, 14], dtype=np.float64)))
        pyomo_nlp.set_duals(np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float64))
        y = pyomo_nlp.get_duals()
        self.assertTrue(np.array_equal(y, np.asarray([1, 1, 1, 1, 0, 0], dtype=np.float64)))

        fac = pyomo_nlp.get_obj_factor()
        self.assertEqual(fac, 1)
        pyomo_nlp.set_obj_factor(42)
        self.assertEqual(pyomo_nlp.get_obj_factor(), 42)
        pyomo_nlp.set_obj_factor(1)

        f = pyomo_nlp.evaluate_objective()
        self.assertEqual(f, 900)

        gradf = pyomo_nlp.evaluate_grad_objective()
        comparison_gradf = np.asarray([0, 0, 0, 0, 0, 0, 60, 0, 0], dtype=np.float64)
        check_vectors_specific_order(self, gradf, x_order, comparison_gradf, comparison_x_order)
        c = pyomo_nlp.evaluate_constraints()
        comparison_c = np.asarray([-2, 26, -13, -22, 0, 50], dtype=np.float64)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)
        c = np.zeros(6)
        pyomo_nlp.evaluate_constraints(out=c)
        check_vectors_specific_order(self, c, c_order, comparison_c, comparison_c_order)

        j = pyomo_nlp.evaluate_jacobian()
        comparison_j = np.asarray([[-1,   9,  12,  1, 0,  0,  0,   0,   0],
                                   [ 0,  18,  24, -1, 1,  0,  0,   0,   0],
                                   [ 0,  -9, -12,  1, 0, -1,  0,   0,   0],
                                   [ 1, -36, -48,  0, 0,  0, -1,   0,   0],
                                   [ 1,   0,   0,  0, 0,  0,  0, -10,   0],
                                   [ 0,   0,   0,  0, 0,  0,  1,   0, -10]])
        
        check_sparse_matrix_specific_order(self, j, c_order, x_order, comparison_j, comparison_c_order, comparison_x_order)

        with self.assertRaises(NotImplementedError):
            h = pyomo_nlp.evaluate_hessian_lag()

    def test_external_greybox_solve(self):
        m = pyo.ConcreteModel()
        m.hin = pyo.Var(bounds=(0,None), initialize=10)
        m.hout = pyo.Var(bounds=(0,None))
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(PressureDropTwoEqualitiesTwoOutputs())
        m.incon = pyo.Constraint(expr= 0 <= m.egb.inputs['Pin'] - 10*m.hin)
        m.outcon = pyo.Constraint(expr= 0 == m.egb.outputs['Pout'] - 10*m.hout)
        m.egb.inputs['Pin'].value = 100
        m.egb.inputs['Pin'].setlb(50)
        m.egb.inputs['Pin'].setub(150)
        m.egb.inputs['c'].value = 2
        m.egb.inputs['c'].setlb(1)
        m.egb.inputs['c'].setub(5)
        m.egb.inputs['F'].value = 3
        m.egb.inputs['F'].setlb(1)
        m.egb.inputs['F'].setub(5)
        m.egb.inputs['P1'].value = 80
        m.egb.inputs['P1'].setlb(10)
        m.egb.inputs['P1'].setub(90)
        m.egb.inputs['P3'].value = 70
        m.egb.inputs['P3'].setlb(20)
        m.egb.inputs['P3'].setub(80)
        m.egb.outputs['P2'].value = 75
        m.egb.outputs['P2'].setlb(15)
        m.egb.outputs['P2'].setub(85)
        m.egb.outputs['Pout'].value = 50
        m.egb.outputs['Pout'].setlb(30)
        m.egb.outputs['Pout'].setub(70)
        m.obj = pyo.Objective(expr=(m.egb.outputs['Pout']-20)**2)
        pyomo_nlp = PyomoGreyBoxNLP(m)

        options = {'hessian_approximation':'limited-memory'}
        solver = CyIpoptSolver(CyIpoptNLP(pyomo_nlp), options)
        x, info = solver.solve(tee=True)

        assert False


def check_vectors_specific_order(tst, v1, v1order, v2, v2order):
    tst.assertEqual(len(v1), len(v1order))
    tst.assertEqual(len(v2), len(v2order))
    tst.assertEqual(len(v1), len(v2))
    v2map = {s:i for i,s in enumerate(v2order)}
    for i,s in enumerate(v1order):
        tst.assertEqual(v1[i], v2[v2map[s]])

def check_sparse_matrix_specific_order(tst, m1, m1rows, m1cols, m2, m2rows, m2cols):
    tst.assertEqual(m1.shape[0], len(m1rows))
    tst.assertEqual(m1.shape[1], len(m1cols))
    tst.assertEqual(m2.shape[0], len(m2rows))
    tst.assertEqual(m2.shape[1], len(m2cols))
    tst.assertEqual(len(m1rows), len(m2rows))
    tst.assertEqual(len(m1cols), len(m2cols))

    m1c = m1.todense()
    m2c = np.zeros((len(m2rows), len(m2cols)))
    rowmap = [m2rows.index(x) for x in m1rows]
    colmap = [m2cols.index(x) for x in m1cols]
    for i in range(len(m1rows)):
        for j in range(len(m1cols)):
            m2c[i,j] = m2[rowmap[i], colmap[j]]

    #print(m1c)
    #print(m2c)
    tst.assertTrue(np.array_equal(m1c, m2c))

"""
    def test_interface(self):
        # weird, this is really a test of the test class above
        # but we could add code later, so...
        iom = PressureDropModel()
        iom.set_inputs(np.ones(4))
        o = iom.evaluate_outputs()
        expected_o = np.asarray([0.0, -1.0], dtype=np.float64)
        self.assertTrue(np.array_equal(o, expected_o))

        jac = iom.evaluate_derivatives()
        expected_jac = np.asarray([[1, -1, 0, -2], [1, -1, -1, -4]], dtype=np.float64)
        self.assertTrue(np.array_equal(jac.todense(), expected_jac))

    def test_pyomo_external_model(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var()
        m.P2 = pyo.Var()

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(m,
                                        PressureDropModel(),
                                        [m.Pin, m.c1, m.c2, m.F],
                                        [m.P1, m.P2]
                                        )

        # check that the dummy variable is initialized
        expected_dummy_var_value = pyo.value(m.Pin) + pyo.value(m.c1) + pyo.value(m.c2) + pyo.value(m.F) \
            + 0 + 0
            # + pyo.value(m.P1) + pyo.value(m.P2) # not initialized - therefore should use zero
        self.assertAlmostEqual(pyo.value(m._dummy_variable_CyIpoptPyomoExNLP), expected_dummy_var_value)

        # solve the problem
        solver = CyIpoptSolver(cyipopt_problem, {'hessian_approximation':'limited-memory'})
        x, info = solver.solve(tee=False)
        cyipopt_problem.load_x_into_pyomo(x)
        self.assertAlmostEqual(pyo.value(m.c1), 0.1, places=5)
        self.assertAlmostEqual(pyo.value(m.c2), 0.5, places=5)

    def test_pyomo_external_model_scaling(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var()
        m.P2 = pyo.Var()

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        # set scaling parameters for the pyomo variables and constraints
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.obj] = 0.1 # scale the objective
        m.scaling_factor[m.Pin] = 2.0 # scale the variable
        m.scaling_factor[m.c1] = 3.0 # scale the variable
        m.scaling_factor[m.c2] = 4.0 # scale the variable
        m.scaling_factor[m.F] = 5.0 # scale the variable
        m.scaling_factor[m.P1] = 6.0 # scale the variable
        m.scaling_factor[m.P2] = 7.0 # scale the variable
        m.scaling_factor[m.F_con] = 8.0 # scale the pyomo constraint
        m.scaling_factor[m.Pin_con] = 9.0 # scale the pyomo constraint

        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(pyomo_model=m,
                                        ex_input_output_model=PressureDropModel(),
                                        inputs=[m.Pin, m.c1, m.c2, m.F],
                                        outputs=[m.P1, m.P2],
                                        outputs_eqn_scaling=[10.0, 11.0]
                                        )

        # solve the problem
        options={'hessian_approximation':'limited-memory',
                 'nlp_scaling_method': 'user-scaling',
                 'output_file': '_cyipopt-pyomo-ext-scaling.log',
                 'file_print_level':10,
                 'max_iter': 0}
        solver = CyIpoptSolver(cyipopt_problem, options=options)
        x, info = solver.solve(tee=False)

        with open('_cyipopt-pyomo-ext-scaling.log', 'r') as fd:
            solver_trace = fd.read()
        os.remove('_cyipopt-pyomo-ext-scaling.log')

        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn('output_file = _cyipopt-pyomo-ext-scaling.log', solver_trace)
        self.assertIn('objective scaling factor = 0.1', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
        self.assertIn('x scaling vector[    1]= 6.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    2]= 7.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 2.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    4]= 3.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    5]= 4.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    6]= 5.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    7]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 5 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 8.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    2]= 9.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    3]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    4]= 1.0000000000000000e+01', solver_trace)
        self.assertIn('c scaling vector[    5]= 1.1000000000000000e+01', solver_trace)

    def test_pyomo_external_model_ndarray_scaling(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var()
        m.P2 = pyo.Var()

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        # set scaling parameters for the pyomo variables and constraints
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.obj] = 0.1 # scale the objective
        m.scaling_factor[m.Pin] = 2.0 # scale the variable
        m.scaling_factor[m.c1] = 3.0 # scale the variable
        m.scaling_factor[m.c2] = 4.0 # scale the variable
        m.scaling_factor[m.F] = 5.0 # scale the variable
        m.scaling_factor[m.P1] = 6.0 # scale the variable
        m.scaling_factor[m.P2] = 7.0 # scale the variable
        m.scaling_factor[m.F_con] = 8.0 # scale the pyomo constraint
        m.scaling_factor[m.Pin_con] = 9.0 # scale the pyomo constraint

        # test that this all works with ndarray input as well
        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(pyomo_model=m,
                                        ex_input_output_model=PressureDropModel(),
                                        inputs=[m.Pin, m.c1, m.c2, m.F],
                                        outputs=[m.P1, m.P2],
                                        outputs_eqn_scaling=np.asarray([10.0, 11.0], dtype=np.float64)
                                        )

        # solve the problem
        options={'hessian_approximation':'limited-memory',
                 'nlp_scaling_method': 'user-scaling',
                 'output_file': '_cyipopt-pyomo-ext-scaling-ndarray.log',
                 'file_print_level':10,
                 'max_iter': 0}
        solver = CyIpoptSolver(cyipopt_problem, options=options)
        x, info = solver.solve(tee=False)

        with open('_cyipopt-pyomo-ext-scaling-ndarray.log', 'r') as fd:
            solver_trace = fd.read()
        os.remove('_cyipopt-pyomo-ext-scaling-ndarray.log')

        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn('output_file = _cyipopt-pyomo-ext-scaling-ndarray.log', solver_trace)
        self.assertIn('objective scaling factor = 0.1', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 7 elements:', solver_trace)
        self.assertIn('x scaling vector[    1]= 6.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    2]= 7.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 2.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    4]= 3.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    5]= 4.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    6]= 5.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    7]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 5 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 8.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    2]= 9.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    3]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('c scaling vector[    4]= 1.0000000000000000e+01', solver_trace)
        self.assertIn('c scaling vector[    5]= 1.1000000000000000e+01', solver_trace)

    def test_pyomo_external_model_dummy_var_initialization(self):
        m = pyo.ConcreteModel()
        m.Pin = pyo.Var(initialize=100, bounds=(0,None))
        m.c1 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.c2 = pyo.Var(initialize=1.0, bounds=(0,None))
        m.F = pyo.Var(initialize=10, bounds=(0,None))

        m.P1 = pyo.Var(initialize=75.0)
        m.P2 = pyo.Var(initialize=50.0)

        m.F_con = pyo.Constraint(expr = m.F == 10)
        m.Pin_con = pyo.Constraint(expr = m.Pin == 100)

        # simple parameter estimation test
        m.obj = pyo.Objective(expr= (m.P1 - 90)**2 + (m.P2 - 40)**2)

        cyipopt_problem = \
            PyomoExternalCyIpoptProblem(m,
                                        PressureDropModel(),
                                        [m.Pin, m.c1, m.c2, m.F],
                                        [m.P1, m.P2]
                                        )

        # check that the dummy variable is initialized
        expected_dummy_var_value = pyo.value(m.Pin) + pyo.value(m.c1) + pyo.value(m.c2) + pyo.value(m.F) \
            + pyo.value(m.P1) + pyo.value(m.P2)
        self.assertAlmostEqual(pyo.value(m._dummy_variable_CyIpoptPyomoExNLP), expected_dummy_var_value)
        # check that the dummy constraint is satisfied
        self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body),pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.lower))
        self.assertAlmostEqual(pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.body),pyo.value(m._dummy_constraint_CyIpoptPyomoExNLP.upper))

        # solve the problem
        solver = CyIpoptSolver(cyipopt_problem, {'hessian_approximation':'limited-memory'})
        x, info = solver.solve(tee=False)
        cyipopt_problem.load_x_into_pyomo(x)
        self.assertAlmostEqual(pyo.value(m.c1), 0.1, places=5)
        self.assertAlmostEqual(pyo.value(m.c2), 0.5, places=5)

"""
if __name__ == '__main__':
    TestPyomoGreyBoxNLP().test_external_greybox_solve(self)
