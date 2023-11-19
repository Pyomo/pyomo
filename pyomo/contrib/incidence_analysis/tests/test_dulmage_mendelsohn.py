#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
from pyomo.contrib.incidence_analysis.interface import get_structural_incidence_matrix
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_dynamic_model,
)

import pyomo.common.unittest as unittest


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionDMMatrixInterface(unittest.TestCase):
    def test_square_well_posed_model(self):
        N = 4
        m = make_gas_expansion_model(N)
        m.F[0].fix()
        m.rho[0].fix()
        m.T[0].fix()

        variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
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

        variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
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
        matched_con_set = set(
            con_idx_map[con] for con in constraints if con is not m.ideal_gas[0]
        )
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
        self.assertEqual(M, 4 * N_model + 1)
        self.assertEqual(N, 4 * (N_model + 1))

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

    def test_recover_matching(self):
        N_model = 4
        m = make_gas_expansion_model(N_model)
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(variables, constraints)
        rdmp, cdmp = dulmage_mendelsohn(imat)
        rmatch = rdmp.underconstrained + rdmp.square + rdmp.overconstrained
        cmatch = cdmp.underconstrained + cdmp.square + cdmp.overconstrained
        matching = list(zip(rmatch, cmatch))
        rmatch = [r for (r, c) in matching]
        cmatch = [c for (r, c) in matching]
        # Assert that the matched rows and columns contain no duplicates
        self.assertEqual(len(set(rmatch)), len(rmatch))
        self.assertEqual(len(set(cmatch)), len(cmatch))
        entry_set = set(zip(imat.row, imat.col))
        for i, j in matching:
            # Assert that each pair in the matching is a valid entry
            # in the matrix
            self.assertIn((i, j), entry_set)


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestDynamicModel(unittest.TestCase):
    def test_rectangular_model(self):
        m = make_dynamic_model()

        m.height[0].fix()

        variables = [v for v in m.component_data_objects(pyo.Var) if not v.fixed]
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
        self.assertEqual(len(row_partition[2]), M - 1)
        row_indices = set([i for i in range(M) if i != con_idx_map[m.flow_out_eqn[0]]])
        self.assertEqual(set(row_partition[2]), row_indices)

        # Potentially unmatched columns
        self.assertEqual(len(col_partition[0]), N - M)
        self.assertEqual(len(col_partition[1]), M - 1)
        potentially_unmatched = col_partition[0] + col_partition[1]
        col_indices = set([i for i in range(N) if i != var_idx_map[m.flow_out[0]]])
        self.assertEqual(set(potentially_unmatched), col_indices)


if __name__ == "__main__":
    unittest.main()
