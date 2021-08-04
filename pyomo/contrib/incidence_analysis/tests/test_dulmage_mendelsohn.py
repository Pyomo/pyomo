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
        get_structural_incidence_matrix,
        )
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
        dulmage_mendelsohn,
        )

import pyomo.common.unittest as unittest


def make_gas_expansion_model(N=2):
    """
    This is the simplest model I could think of that has a
    subsystem with a non-trivial block triangularization.
    Something like a gas (somehow) undergoing a series
    of isentropic expansions.
    """
    m = pyo.ConcreteModel()
    m.streams = pyo.Set(initialize=range(N + 1))
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


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionDMMatrixInterface(unittest.TestCase):

    def test_square_well_posed_model(self):
        N = 4
        m = make_gas_expansion_model(N)
        m.F[0].fix()
        m.rho[0].fix()
        m.T[0].fix()

        variables = [v for v in m.component_data_objects(pyo.Var)
                if not v.fixed]
        constraints = list(m.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(variables, constraints)

        N, M = imat.shape
        self.assertEqual(N, M)

        row_partition, col_partition = dulmage_mendelsohn(imat)

        # Unmatched, reachable-from-unmatched, and matched-with-reachable
        # subsets are all empty.
        self.assertEqual(len(row_partition[0]), 0)
        self.assertEqual(len(row_partition[1]), 0)
        self.assertEqual(len(row_partition[2]), 0)
        self.assertEqual(len(col_partition[0]), 0)
        self.assertEqual(len(col_partition[1]), 0)
        self.assertEqual(len(col_partition[2]), 0)

        # All nodes belong to the "well-determined" subset
        self.assertEqual(len(row_partition[3]), M)
        self.assertEqual(len(col_partition[3]), N)

    def test_square_ill_posed_model(self):
        N = 1
        m = make_gas_expansion_model(N)
        m.P[0].fix()
        m.rho[0].fix()
        m.T[0].fix()

        variables = [v for v in m.component_data_objects(pyo.Var)
                if not v.fixed]
        constraints = list(m.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(variables, constraints)

        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        N, M = imat.shape
        self.assertEqual(N, M)

        row_partition, col_partition = dulmage_mendelsohn(imat)

        # Only unmatched constraint is ideal_gas[0]
        unmatched_rows = [con_idx_map[m.ideal_gas[0]]]
        self.assertEqual(row_partition[0], unmatched_rows)
        # No other constraints can possibly be unmatched.
        self.assertEqual(row_partition[1], [])
        # The potentially unmatched variables have four constraints
        # between them
        matched_con_set = set(con_idx_map[con] for con in constraints
                if con is not m.ideal_gas[0])
        self.assertEqual(set(row_partition[2]), matched_con_set)

        # All variables are potentially unmatched
        potentially_unmatched_set = set(range(len(variables)))
        potentially_unmatched = col_partition[0] + col_partition[1]
        self.assertEqual(set(potentially_unmatched), potentially_unmatched_set)

    def test_rectangular_system(self):
        N_model = 2
        m = make_gas_expansion_model(N_model)
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        M, N = imat.shape
        self.assertEqual(M, 4*N_model + 1)
        self.assertEqual(N, 4*(N_model + 1))

        row_partition, col_partition = dulmage_mendelsohn(imat)

        # No unmatched constraints
        self.assertEqual(row_partition[0], [])
        self.assertEqual(row_partition[1], [])

        # All constraints are matched with potentially unmatched variables
        matched_con_set = set(range(len(constraints)))
        self.assertEqual(set(row_partition[2]), matched_con_set)

        # 3 unmatched variables
        self.assertEqual(len(col_partition[0]), 3)
        # All variables are potentially unmatched
        potentially_unmatched = col_partition[0] + col_partition[1]
        potentially_unmatched_set = set(range(len(variables)))
        self.assertEqual(set(potentially_unmatched), potentially_unmatched_set)


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestDynamicModel(unittest.TestCase):

    def test_rectangular_model(self):
        m = make_dynamic_model()

        m.height[0].fix()

        variables = [v for v in m.component_data_objects(pyo.Var)
                if not v.fixed]
        constraints = list(m.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        M, N = imat.shape
        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        row_partition, col_partition = dulmage_mendelsohn(imat)

        # No unmatched rows
        self.assertEqual(row_partition[0], [])
        self.assertEqual(row_partition[1], [])

        # Assert that the square subsystem contains the components we expect
        self.assertEqual(len(row_partition[3]), 1)
        self.assertEqual(row_partition[3][0], con_idx_map[m.flow_out_eqn[0]])

        self.assertEqual(len(col_partition[3]), 1)
        self.assertEqual(col_partition[3][0], var_idx_map[m.flow_out[0]])

        # Assert that underdetermined subsystem contains the components
        # we expect

        # Rows matched with potentially unmatched columns
        self.assertEqual(len(row_partition[2]), M-1)
        row_indices = set([i for i in range(M)
            if i != con_idx_map[m.flow_out_eqn[0]]])
        self.assertEqual(set(row_partition[2]), row_indices)

        # Potentially unmatched columns
        self.assertEqual(len(col_partition[0]), N - M)
        self.assertEqual(len(col_partition[1]), M - 1)
        potentially_unmatched = col_partition[0] + col_partition[1]
        col_indices = set([i for i in range(N)
            if i != var_idx_map[m.flow_out[0]]])
        self.assertEqual(set(potentially_unmatched), col_indices)


if __name__ == "__main__":
    unittest.main()
