#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
        IncidenceGraphInterface,
        get_structural_incidence_matrix,
        get_numeric_incidence_matrix,
        )
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
from pyomo.contrib.incidence_analysis.util import (
        generate_strongly_connected_components,
        solve_strongly_connected_components,
        )
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import pyomo.common.unittest as unittest


def make_gas_expansion_model(N=2):
    """
    This is the simplest model I could think of that has a
    subsystem with a non-trivial block triangularization.
    Something like a gas (somehow) undergoing a series
    of isentropic expansions.
    """
    m = pyo.ConcreteModel()
    m.streams = pyo.Set(initialize=range(N+1))
    m.rho = pyo.Var(m.streams, initialize=1)
    m.P = pyo.Var(m.streams, initialize=1)
    m.F = pyo.Var(m.streams, initialize=1)
    m.T = pyo.Var(m.streams, initialize=1)

    m.R = pyo.Param(initialize=8.31)
    m.Q = pyo.Param(m.streams, initialize=1)
    m.gamma = pyo.Param(initialize=1.4*m.R.value)

    def mbal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.rho[i-1]*m.F[i-1] - m.rho[i]*m.F[i] == 0
    m.mbal = pyo.Constraint(m.streams, rule=mbal)

    def ebal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return (
                    m.rho[i-1]*m.F[i-1]*m.T[i-1] +
                    m.Q[i] -
                    m.rho[i]*m.F[i]*m.T[i] == 0
                    )
    m.ebal = pyo.Constraint(m.streams, rule=ebal)

    def expansion(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.P[i]/m.P[i-1] - (m.rho[i]/m.rho[i-1])**m.gamma == 0
    m.expansion = pyo.Constraint(m.streams, rule=expansion)

    def ideal_gas(m, i):
        return m.P[i] - m.rho[i]*m.R*m.T[i] == 0
    m.ideal_gas = pyo.Constraint(m.streams, rule=ideal_gas)

    return m


def make_dynamic_model(**disc_args):
    # Level control model
    m = pyo.ConcreteModel()
    m.time = dae.ContinuousSet(initialize=[0.0, 10.0])
    m.height = pyo.Var(m.time, initialize=1.0)
    m.flow_in = pyo.Var(m.time, initialize=1.0)
    m.flow_out = pyo.Var(m.time, initialize=0.5)
    m.dhdt = dae.DerivativeVar(m.height, wrt=m.time, initialize=0.0)

    m.area = pyo.Param(initialize=1.0)
    m.flow_const = pyo.Param(initialize=0.5)

    def diff_eqn_rule(m, t):
        return m.area*m.dhdt[t] - (m.flow_in[t] - m.flow_out[t]) == 0
    m.diff_eqn = pyo.Constraint(m.time, rule=diff_eqn_rule)

    def flow_out_rule(m, t):
        return m.flow_out[t] - (m.flow_const*pyo.sqrt(m.height[t])) == 0
    m.flow_out_eqn = pyo.Constraint(m.time, rule=flow_out_rule)

    default_disc_args = {
            "wrt": m.time,
            "nfe": 5,
            "scheme": "BACKWARD",
            }
    default_disc_args.update(disc_args)

    discretizer = pyo.TransformationFactory("dae.finite_difference")
    discretizer.apply_to(m, **default_disc_args)

    return m


class TestGenerateSCC(unittest.TestCase):

    def test_gas_expansion(self):
        N = 5
        m = make_gas_expansion_model(N)
        m.rho[0].fix()
        m.F[0].fix()
        m.T[0].fix()

        self.assertEqual(
                len(list(generate_strongly_connected_components(m))),
                N+1,
                )
        for i, block in enumerate(generate_strongly_connected_components(m)):
            if i == 0:
                # P[0], ideal_gas[0]
                self.assertEqual(len(block.vars), 1)
                self.assertEqual(len(block.cons), 1)

                var_set = ComponentSet([m.P[i]])
                con_set = ComponentSet([m.ideal_gas[i]])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)

                # Other variables are fixed; not included
                self.assertEqual(len(block.input_vars), 0)

            elif i == 1:
                # P[1], rho[1], F[1], T[1], etc.
                self.assertEqual(len(block.vars), 4)
                self.assertEqual(len(block.cons), 4)

                var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                con_set = ComponentSet([
                    m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]
                    ])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)

                # P[0] is in expansion[1]
                other_var_set = ComponentSet([m.P[i-1]])
                self.assertEqual(len(block.input_vars), 1)
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)

            else:
                # P[i], rho[i], F[i], T[i], etc.
                self.assertEqual(len(block.vars), 4)
                self.assertEqual(len(block.cons), 4)

                var_set = ComponentSet([m.P[i], m.rho[i], m.F[i], m.T[i]])
                con_set = ComponentSet([
                    m.ideal_gas[i], m.mbal[i], m.ebal[i], m.expansion[i]
                    ])
                for var, con in zip(block.vars[:], block.cons[:]):
                    self.assertIn(var, var_set)
                    self.assertIn(con, con_set)

                # P[i-1], rho[i-1], F[i-1], T[i-1], etc.
                other_var_set = ComponentSet([
                    m.P[i-1], m.rho[i-1], m.F[i-1], m.T[i-1]
                    ])
                self.assertEqual(len(block.input_vars), 4)
                for var in block.input_vars[:]:
                    self.assertIn(var, other_var_set)

    def test_dynamic_backward_disc(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme="BACKWARD")
        time = m.time
        t0 = m.time.first()

        m.flow_in.fix()
        m.height[t0].fix()

        time.pprint()

        self.assertEqual(
                len(list(generate_strongly_connected_components(m))),
                nfe+2,
                # The "initial constraints" have two SCCs because they
                # decompose into the algebraic equation and differential
                # equation. This decomposition is because the discretization
                # equation is not present.
                )
        for i, block in enumerate(generate_strongly_connected_components(m)):
            t = m.time[i+1] # Pyomo sets are base-1-indexed...
            if i != 0:
                t_prev = m.time.prev(t)

            if i == 0:
                con_set = ComponentSet([m.flow_out_eqn[t]])


if __name__ == "__main__":
    unittest.main()
