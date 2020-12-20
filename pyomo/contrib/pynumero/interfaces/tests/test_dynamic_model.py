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
import math

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_sparse as spa, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptSolver, CyIpoptNLP, ipopt, ipopt_available,
)

from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
from ..pyomo_nlp import PyomoGreyBoxNLP

A1 = 5
A2 = 10
c1 = 3
c2 = 4
Fin = 2
Tf = 10
dt = 1
def create_pyomo_model():
    m = pyo.ConcreteModel()
    
    # timesteps
    m.T = pyo.Set(initialize=list(range(Tf)), ordered=True)
    m.Tu = pyo.Set(initialize=list(range(Tf))[1:], ordered=True)
    
    # inputs (controls)
    m.F1 = pyo.Var(m.Tu, bounds=(0,5), initialize=1.0)
    m.F2 = pyo.Var(m.Tu, bounds=(0,5), initialize=1.0)

    # state variables
    m.h1 = pyo.Var(m.T, bounds=(0,None), initialize=1.0)
    m.h2 = pyo.Var(m.T, bounds=(0,None), initialize=1.0)

#    # algebraics (outputs)
#    m.F12 = pyo.Var(m.T, bounds=(0,None), initialize=1.0)
#    m.Fo = pyo.Var(m.T, bounds=(0,None), initialize=1.0)

    @m.Constraint(m.Tu)
    def h1bal(m, t):
        return A1/dt * (m.h1[t] - m.h1[t-1]) - m.F1[t] + c1*pyo.sqrt(m.h1[t]) == 0

    @m.Constraint(m.Tu)
    def h2bal(m, t):
        return A2/dt * (m.h2[t] - m.h2[t-1]) - c1*pyo.sqrt(m.h1[t]) - m.F2[t] + c2*pyo.sqrt(m.h2[t]) == 0

#    @m.Constraint(m.T)
#    def F12con(m, t):
#        return c1*pyo.sqrt(m.h1[t]) - m.F12[t] == 0
#
#    @m.Constraint(m.T)
#    def Focon(m, t):
#        return c2*pyo.sqrt(m.h2[t]) - m.Fo[t] == 0

    @m.Constraint(m.Tu)
    def min_inflow(m, t):
        return 2 <= m.F1[t] + m.F2[t]

    @m.Constraint(m.Tu)
    def max_inflow(m, t):
        return m.F1[t] + m.F2[t] <= 5

    m.h10 = pyo.Constraint( expr=m.h1[m.T.first()] == 2.0 )
    m.h20 = pyo.Constraint( expr=m.h2[m.T.first()] == 2.0 )
    m.obj = pyo.Objective( expr= sum((m.h1[t]-1.0)**2  + (m.h2[t]-1.5)**2 for t in m.T) )
    
    return m

"""
class TwoTanksSeries(ExternalGreyBoxModel):
    def __init__(self, tstart, tend):
        self._N = tend - tstart + 1
        self._input_names = ['Fin_{}'.format(t) for t in range(tstart,tend+1)]
        self._input_names.extend(['h1_{}'.format(t) for t in range(tstart,tend+1)])
        self._input_names.extend(['h2_{}'.format(t) for t in range(tstart,tend+1)])
        self._output_names = ['F12_{}'.format(t) for t in range(tstart,tend+1)]
        self._output_names.extend(['Fo_{}'.format(t) for t in range(tstart,tend+1)])
        self._equality_constraint_names = ['h1bal_{}'.format(t) for t in range(tstart+1,tend+1)]
        self._equality_constraint_names.extend(['h2bal_{}'.format(t) for t in range(tstart+1,tend+1)])
        self._Fin = np.zeros(self._N)
        self._h1 = np.zeros(self._N)
        self._h2 = np.zeros(self._N)
        self._eq_con_mult_values = np.zeros(2*self._N-2)
        self._output_con_mult_values = np.zeros(2*self._N)

    def model_capabilities(self):
        capabilities = self.ModelCapabilities()
        capabilities.supports_jacobian_equality_constraints = True
        capabilities.supports_hessian_equality_constraints = True
        capabilities.supports_jacobian_outputs = True
        capabilities.supports_hessian_outputs = True
        return capabilities

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def set_input_values(self, input_values):
        assert len(input_values) == 3*self._N
        np.copyto(self._Fin,input_values[:self._N])
        np.copyto(self._h1, input_values[self._N:2*self._N])
        np.copyto(self._h2, input_values[2*self._N:3*self._N])

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2*(self._N-1)
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)
        
    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 2*self._N
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_equality_constraints(self):
        Fin = self._Fin
        h1 = self._h1
        h2 = self._h2
        
        resid = np.zeros(2*self._N)

        for t in range(1,self._N):
            resid[t-1] = A1/dt*(h1[t]-h1[t-1]) - Fin[t] + c1*math.sqrt(h1[t])

        for t in range(1,self._N):
            resid[t-1+self._N] = A2/dt*(h2[t]-h2[t-1]) - c1*math.sqrt(h1[t]) + c2*math.sqrt(h2[t])

        return resid

    def evaluate_outputs(self):
        Fin = self._Fin
        h1 = self._h1
        h2 = self._h2
        
        resid = np.zeros(2*self._N)

        for t in range(0,self._N):
            resid[t] = c1*math.sqrt(h1[t])

        for t in range(0,self._N):
            resid[t+self._N] = c2*math.sqrt(h2[t])

        return resid

    def evaluate_jacobian_equality_constraints(self):
        pass
    
    def evaluate_jacobian_outputs(self):
        pass

    def evaluate_hessian_equality_constraints(self):
        pass
    
    def evaluate_hessian_outputs(self):
        pass
"""

class TestGreyBoxModel(unittest.TestCase):
    def test_full_pyomo(self):
        m = create_pyomo_model()
        solver = pyo.SolverFactory('ipopt')
        status = solver.solve(m, tee=True)
        m.pprint()
        m.display()
        assert False
"""
        m2 = pyo.ConcreteModel()
        m2.egb = ExternalGreyBoxBlock()
        m2.egb.set_external_model(TwoTanksSeries(0,Tf-1))
        m2.egb.pprint()

        print('foo')

        # let's assign inputs / outputs and check the equations
        N = Tf
        u = np.zeros(3*N)
        for t in range(N):
            u[t] = pyo.value(m.Fin[t])
            u[t+N] = pyo.value(m.h1[t])
            u[t+2*N] = pyo.value(m.h2[t])
        print(u)

        ex_model = m2.egb._ex_model
        ex_model.set_input_values(u)
        resid = ex_model.evaluate_equality_constraints()
        print(resid)
        osoln = np.zeros(len(m.F12)+len(m.Fo))
        for i in m.T:
            osoln[i] = pyo.value(m.F12[i])
        for i in m.T:
            osoln[i+N] = pyo.value(m.Fo[i])
        o = ex_model.evaluate_outputs()
        print(o)
        print(osoln)
        assert False

    def test_evaluations(self):
        u = np.asarray([0.00000000e+00, 1.50357444e-08, 3.60100604e-08, 1.10971623e-07,
                        3.71388285e+00, 4.99999960e+00, 4.99999957e+00, 4.99999927e+00,
                        4.99999816e+00, 4.99999025e+00, 2.00000000e+00, 1.31259006e+00,
                        7.82004098e-01, 4.01717155e-01, 6.57847087e-01, 1.04461002e+00,
                        1.34799167e+00, 1.59114738e+00, 1.78869451e+00, 1.95068994e+00,
                        2.00000000e+00, 1.80613490e+00, 1.57019776e+00, 1.30363406e+00,
                        1.12305939e+00, 1.02475680e+00, 9.77576129e-01, 9.63388730e-01,
                        9.70549167e-01, 9.91295233e-01])

        o = np.asarray([4.24264069, 3.43704969, 2.65292987, 1.90143483, 2.4332332,  3.06618495,
                        3.4830913,  3.78422072, 4.01226253, 4.19001307, 5.65685425, 5.37570073,
                        5.01230128, 4.56707181, 4.23897986, 4.04921088, 3.95489798, 3.92609471,
                        3.94065815, 3.98255241])
        
        m2 = pyo.ConcreteModel()
        m2.egb = ExternalGreyBoxBlock()
        m2.egb.set_external_model(TwoTanksSeries(0,Tf-1))
        m2.obj = pyo.Objective(expr=sum( (m2.egb.inputs['h2_{}'.format(t)] - 1.0)**2 for t in range(Tf) ) )

        pnlp = PyomoGreyBoxNLP(m2)
        assert False
        
        ex_model = m2.egb._ex_model
        ex_model.set_input_values(u)
        resid = ex_model.evaluate_equality_constraints()
        print(resid)
        osoln = np.zeros(len(m.F12)+len(m.Fo))
        for i in m.T:
            osoln[i] = pyo.value(m.F12[i])
        for i in m.T:
            osoln[i+N] = pyo.value(m.Fo[i])
        o = ex_model.evaluate_outputs()

"""


        
