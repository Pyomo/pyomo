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
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    get_structural_incidence_matrix,
    get_numeric_incidence_matrix,
    get_incidence_graph,
    )
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
    dulmage_mendelsohn,
    )
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_degenerate_solid_phase_model,
        )
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
if networkx_available:
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from pyomo.contrib.pynumero.asl import AmplInterface

import pyomo.common.unittest as unittest


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
@unittest.skipUnless(AmplInterface.available(), "pynumero_ASL is not available")
class TestGasExpansionNumericIncidenceMatrix(unittest.TestCase):
    """
    This class tests the get_numeric_incidence_matrix function on
    the gas expansion model.
    """
    def test_incidence_matrix(self):
        N = 5
        model = make_gas_expansion_model(N)
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_numeric_incidence_matrix(all_vars, all_cons)
        n_var = 4*(N+1)
        n_con = 4*N+1
        self.assertEqual(imat.shape, (n_con, n_var))

        var_idx_map = ComponentMap((v, i) for i, v in enumerate(all_vars))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(all_cons))

        # Map constraints to the variables they contain.
        csr_map = ComponentMap()
        csr_map.update((model.mbal[i], ComponentSet([
            model.F[i],
            model.F[i-1],
            model.rho[i],
            model.rho[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.ebal[i], ComponentSet([
            model.F[i],
            model.F[i-1],
            model.rho[i],
            model.rho[i-1],
            model.T[i],
            model.T[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.expansion[i], ComponentSet([
            model.rho[i],
            model.rho[i-1],
            model.P[i],
            model.P[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.ideal_gas[i], ComponentSet([
            model.P[i],
            model.rho[i],
            model.T[i],
            ])) for i in model.streams)

        # Map constraint and variable indices to the values of the derivatives
        # Note that the derivative values calculated here depend on the model's
        # canonical form.
        deriv_lookup = {}
        m = model # for convenience
        for s in model.streams:
            # Ideal gas:
            i = con_idx_map[model.ideal_gas[s]]
            j = var_idx_map[model.P[s]]
            deriv_lookup[i,j] = 1.0

            j = var_idx_map[model.rho[s]]
            deriv_lookup[i,j] = - model.R.value*model.T[s].value

            j = var_idx_map[model.T[s]]
            deriv_lookup[i,j] = - model.R.value*model.rho[s].value

            if s != model.streams.first():
                # Expansion:
                i = con_idx_map[model.expansion[s]]
                j = var_idx_map[model.P[s]]
                deriv_lookup[i,j] = 1/model.P[s-1].value

                j = var_idx_map[model.P[s-1]]
                deriv_lookup[i,j] = -model.P[s].value/model.P[s-1]**2

                j = var_idx_map[model.rho[s]]
                deriv_lookup[i,j] = pyo.value(
                        -m.gamma*(m.rho[s]/m.rho[s-1])**(m.gamma-1)/m.rho[s-1]
                        )

                j = var_idx_map[model.rho[s-1]]
                deriv_lookup[i,j] = pyo.value(
                        -m.gamma*(m.rho[s]/m.rho[s-1])**(m.gamma-1) *
                        (-m.rho[s]/m.rho[s-1]**2)
                        )

                # Energy balance:
                i = con_idx_map[m.ebal[s]]
                j = var_idx_map[m.rho[s-1]]
                deriv_lookup[i,j] = pyo.value(m.F[s-1]*m.T[s-1])

                j = var_idx_map[m.F[s-1]]
                deriv_lookup[i,j] = pyo.value(m.rho[s-1]*m.T[s-1])

                j = var_idx_map[m.T[s-1]]
                deriv_lookup[i,j] = pyo.value(m.F[s-1]*m.rho[s-1])

                j = var_idx_map[m.rho[s]]
                deriv_lookup[i,j] = pyo.value(-m.F[s]*m.T[s])

                j = var_idx_map[m.F[s]]
                deriv_lookup[i,j] = pyo.value(-m.rho[s]*m.T[s])

                j = var_idx_map[m.T[s]]
                deriv_lookup[i,j] = pyo.value(-m.F[s]*m.rho[s])

                # Mass balance:
                i = con_idx_map[m.mbal[s]]
                j = var_idx_map[m.rho[s-1]]
                deriv_lookup[i,j] = pyo.value(m.F[s-1])

                j = var_idx_map[m.F[s-1]]
                deriv_lookup[i,j] = pyo.value(m.rho[s-1])

                j = var_idx_map[m.rho[s]]
                deriv_lookup[i,j] = pyo.value(-m.F[s])

                j = var_idx_map[m.F[s]]
                deriv_lookup[i,j] = pyo.value(-m.rho[s])


        # Want to test that the columns have the rows we expect.
        i = model.streams.first()
        for i, j, e in zip(imat.row, imat.col, imat.data):
            con = all_cons[i]
            var = all_vars[j]
            self.assertIn(var, csr_map[con])
            csr_map[con].remove(var)
            self.assertAlmostEqual(pyo.value(deriv_lookup[i,j]), pyo.value(e), 8)
        # And no additional rows
        for con in csr_map:
            self.assertEqual(len(csr_map[con]), 0)

    #
    # The following tests were copied from the
    # TestGasExpansionModelInterfaceClass test cases, and modified
    # to use the data format returned by get_numeric_incidence_matrix.
    #
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_numeric_incidence_matrix(all_vars, all_cons)

        n_eqn = len(all_cons)
        matching = maximum_matching(imat)
        values = set(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_numeric_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        n_var = len(variables)
        matching = maximum_matching(imat)
        matching = ComponentMap((c, variables[matching[con_idx_map[c]]])
                for c in constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_numeric_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))
        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))

        row_block_map, col_block_map = block_triangularize(imat)
        var_block_map = ComponentMap((v, col_block_map[var_idx_map[v]])
                for v in variables)
        con_block_map = ComponentMap((c, row_block_map[con_idx_map[c]])
                for c in constraints)

        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionStructuralIncidenceMatrix(unittest.TestCase):
    """
    This class tests the get_structural_incidence_matrix function
    on the gas expansion model.
    """
    def test_incidence_matrix(self):
        N = 5
        model = make_gas_expansion_model(N)
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(all_vars, all_cons)
        n_var = 4*(N+1)
        n_con = 4*N+1
        self.assertEqual(imat.shape, (n_con, n_var))

        var_idx_map = ComponentMap((v, i) for i, v in enumerate(all_vars))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(all_cons))

        # Map constraints to the variables they contain.
        csr_map = ComponentMap()
        csr_map.update((model.mbal[i], ComponentSet([
            model.F[i],
            model.F[i-1],
            model.rho[i],
            model.rho[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.ebal[i], ComponentSet([
            model.F[i],
            model.F[i-1],
            model.rho[i],
            model.rho[i-1],
            model.T[i],
            model.T[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.expansion[i], ComponentSet([
            model.rho[i],
            model.rho[i-1],
            model.P[i],
            model.P[i-1],
            ])) for i in model.streams if i != model.streams.first())
        csr_map.update((model.ideal_gas[i], ComponentSet([
            model.P[i],
            model.rho[i],
            model.T[i],
            ])) for i in model.streams)

        # Want to test that the columns have the rows we expect.
        i = model.streams.first()
        for i, j, e in zip(imat.row, imat.col, imat.data):
            con = all_cons[i]
            var = all_vars[j]
            self.assertIn(var, csr_map[con])
            csr_map[con].remove(var)
            self.assertEqual(e, 1.0)
        # And no additional rows
        for con in csr_map:
            self.assertEqual(len(csr_map[con]), 0)

    #
    # The following tests were copied from the
    # TestGasExpansionModelInterfaceClass test cases, and modified
    # to use the data format returned by get_structural_incidence_matrix.
    #
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        all_vars = list(model.component_data_objects(pyo.Var))
        all_cons = list(model.component_data_objects(pyo.Constraint))
        imat = get_structural_incidence_matrix(all_vars, all_cons)

        n_eqn = len(all_cons)
        matching = maximum_matching(imat)
        values = set(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        n_var = len(variables)
        matching = maximum_matching(imat)
        matching = ComponentMap((c, variables[matching[con_idx_map[c]]])
                for c in constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))
        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))

        row_block_map, col_block_map = block_triangularize(imat)
        var_block_map = ComponentMap((v, col_block_map[var_idx_map[v]])
                for v in variables)
        con_block_map = ComponentMap((c, row_block_map[con_idx_map[c]])
                for c in constraints)

        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
@unittest.skipUnless(AmplInterface.available(), "pynumero_ASL is not available")
class TestGasExpansionModelInterfaceClassNumeric(unittest.TestCase):
    # In these tests, we pass the interface a PyomoNLP and cache
    # its Jacobian.
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(model)
        igraph = IncidenceGraphInterface(nlp)

        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(model)
        igraph = IncidenceGraphInterface(nlp)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        model.obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(model)
        igraph = IncidenceGraphInterface(nlp)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.block_triangularize(
                variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)

    def test_exception(self):
        model = make_gas_expansion_model()
        model.obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(model)
        igraph = IncidenceGraphInterface(nlp)

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionModelInterfaceClassStructural(unittest.TestCase):
    # In these tests we pass a model to the interface and are caching a
    # structural incidence matrix.
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)

        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.block_triangularize(
                variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)

    def test_triangularize_submatrix(self):
        # This test exercises the extraction of a somewhat nontrivial
        # submatrix from a cached incidence matrix.
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of a square,
        # nonsingular subsystem
        variables = []
        half = N//2
        variables.extend(model.P[i] for i in model.streams if i >= half)
        variables.extend(model.T[i] for i in model.streams if i > half)
        variables.extend(model.rho[i] for i in model.streams if i > half)
        variables.extend(model.F[i] for i in model.streams if i > half)

        constraints = []
        constraints.extend(model.ideal_gas[i] for i in model.streams
                if i >= half)
        constraints.extend(model.expansion[i] for i in model.streams
                if i > half)
        constraints.extend(model.mbal[i] for i in model.streams
                if i > half)
        constraints.extend(model.ebal[i] for i in model.streams
                if i > half)

        var_block_map, con_block_map = igraph.block_triangularize(
                variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), (N-half)+1)
        self.assertEqual(len(con_values), (N-half)+1)

        self.assertEqual(var_block_map[model.P[half]], 0)

        for i in model.streams:
            if i > half:
                idx = i - half
                self.assertEqual(var_block_map[model.rho[i]], idx)
                self.assertEqual(var_block_map[model.T[i]], idx)
                self.assertEqual(var_block_map[model.P[i]], idx)
                self.assertEqual(var_block_map[model.F[i]], idx)

                self.assertEqual(con_block_map[model.ideal_gas[i]], idx)
                self.assertEqual(con_block_map[model.expansion[i]], idx)
                self.assertEqual(con_block_map[model.mbal[i]], idx)
                self.assertEqual(con_block_map[model.ebal[i]], idx)

    def test_exception(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

    def test_remove(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)

        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

        variable_set = ComponentSet(igraph.variables)
        self.assertIn(model.F[0], variable_set)
        self.assertIn(model.F[2], variable_set)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        underconstrained_set = ComponentSet(
            var_dmp.unmatched + var_dmp.underconstrained
        )
        self.assertIn(model.F[0], underconstrained_set)
        self.assertIn(model.F[2], underconstrained_set)

        N, M = igraph.incidence_matrix.shape

        # Say we know that these variables and constraints should
        # be matched...
        vars_to_remove = [model.F[0], model.F[2]]
        cons_to_remove = (model.mbal[1], model.mbal[2])
        igraph.remove_nodes(vars_to_remove, cons_to_remove)
        variable_set = ComponentSet(igraph.variables)
        self.assertNotIn(model.F[0], variable_set)
        self.assertNotIn(model.F[2], variable_set)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        underconstrained_set = ComponentSet(
            var_dmp.unmatched + var_dmp.underconstrained
        )
        self.assertNotIn(model.F[0], underconstrained_set)
        self.assertNotIn(model.F[2], underconstrained_set)

        N_new, M_new = igraph.incidence_matrix.shape
        self.assertEqual(N_new, N - len(cons_to_remove))
        self.assertEqual(M_new, M - len(vars_to_remove))


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestGasExpansionModelInterfaceClassNoCache(unittest.TestCase):
    # In these tests we do not cache anything and use the interface
    # simply as a convenient wrapper around the analysis functions,
    # which act on matrices.
    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface()

        constraints = list(model.component_data_objects(pyo.Constraint))
        variables = list(model.component_data_objects(pyo.Var))
        n_eqn = len(constraints)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)

        # The subset of variables and equations we have identified
        # do not have a unique perfect matching. But we at least know
        # this much.
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.block_triangularize(
                variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N+1)
        self.assertEqual(len(con_values), N+1)

        self.assertEqual(var_block_map[model.P[0]], 0)

        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)

                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)

    def test_diagonal_blocks(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, constraints
        )
        self.assertIs(igraph.row_block_map, None)
        self.assertIs(igraph.col_block_map, None)
        self.assertEqual(len(var_blocks), N+1)
        self.assertEqual(len(con_blocks), N+1)

        for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
            var_set = ComponentSet(vars)
            con_set = ComponentSet(cons)

            if i == 0:
                pred_var_set = ComponentSet([model.P[0]])
                self.assertEqual(pred_var_set, var_set)
                pred_con_set = ComponentSet([model.ideal_gas[0]])
                self.assertEqual(pred_con_set, con_set)

            else:
                pred_var_set = ComponentSet([
                    model.rho[i], model.T[i], model.P[i], model.F[i]
                ])
                pred_con_set = ComponentSet([
                    model.ideal_gas[i],
                    model.expansion[i],
                    model.mbal[i],
                    model.ebal[i],
                ])
                self.assertEqual(pred_var_set, var_set)
                self.assertEqual(pred_con_set, con_set)

    def test_diagonal_blocks_with_cached_maps(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(model.T[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.rho[i] for i in model.streams
                if i != model.streams.first())
        variables.extend(model.F[i] for i in model.streams
                if i != model.streams.first())

        constraints = list(model.component_data_objects(pyo.Constraint))

        igraph.block_triangularize(variables, constraints)
        var_blocks, con_blocks = igraph.get_diagonal_blocks(
            variables, constraints
        )
        self.assertIsNot(igraph.row_block_map, None)
        self.assertIsNot(igraph.col_block_map, None)
        self.assertEqual(len(var_blocks), N+1)
        self.assertEqual(len(con_blocks), N+1)

        for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
            var_set = ComponentSet(vars)
            con_set = ComponentSet(cons)

            if i == 0:
                pred_var_set = ComponentSet([model.P[0]])
                self.assertEqual(pred_var_set, var_set)
                pred_con_set = ComponentSet([model.ideal_gas[0]])
                self.assertEqual(pred_con_set, con_set)

            else:
                pred_var_set = ComponentSet([
                    model.rho[i], model.T[i], model.P[i], model.F[i]
                ])
                pred_con_set = ComponentSet([
                    model.ideal_gas[i],
                    model.expansion[i],
                    model.mbal[i],
                    model.ebal[i],
                ])
                self.assertEqual(pred_var_set, var_set)
                self.assertEqual(pred_con_set, con_set)


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestDulmageMendelsohnInterface(unittest.TestCase):

    def test_degenerate_solid_phase_model(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))

        igraph = IncidenceGraphInterface()
        var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)

        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())

        self.assertEqual(len(var_dmp[0]+var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0]+var_dmp[1]:
            self.assertIn(var, underconstrained_vars)

        self.assertEqual(len(con_dmp[2]), len(underconstrained_cons))
        for con in con_dmp[2]:
            self.assertIn(con, underconstrained_cons)

        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)

        self.assertEqual(len(var_dmp[2]), len(overconstrained_vars))
        for var in var_dmp[2]:
            self.assertIn(var, overconstrained_vars)

        self.assertEqual(len(con_dmp[0]+con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0]+con_dmp[1]:
            self.assertIn(con, overconstrained_cons)

    def test_named_tuple(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))

        igraph = IncidenceGraphInterface()
        var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)

        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())

        dmp_vars_under = var_dmp.unmatched + var_dmp.underconstrained
        dmp_vars_over = var_dmp.overconstrained
        dmp_cons_under = con_dmp.underconstrained
        dmp_cons_over = con_dmp.unmatched + con_dmp.overconstrained

        self.assertEqual(len(dmp_vars_under), len(underconstrained_vars))
        for var in dmp_vars_under:
            self.assertIn(var, underconstrained_vars)

        self.assertEqual(len(dmp_cons_under), len(underconstrained_cons))
        for con in dmp_cons_under:
            self.assertIn(con, underconstrained_cons)

        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)

        self.assertEqual(len(dmp_vars_over), len(overconstrained_vars))
        for var in dmp_vars_over:
            self.assertIn(var, overconstrained_vars)

        self.assertEqual(len(dmp_cons_over), len(overconstrained_cons))
        for con in dmp_cons_over:
            self.assertIn(con, overconstrained_cons)

    def test_incidence_graph(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        graph = get_incidence_graph(variables, constraints)
        matrix = get_structural_incidence_matrix(variables, constraints)
        from_matrix = from_biadjacency_matrix(matrix)

        self.assertEqual(graph.nodes, from_matrix.nodes)
        self.assertEqual(graph.edges, from_matrix.edges)

    def test_dm_graph_interface(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        graph = get_incidence_graph(variables, constraints)

        M, N = len(constraints), len(variables)

        top_nodes = list(range(M))
        con_dmp, var_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)
        con_dmp = tuple([constraints[i] for i in subset] for subset in con_dmp)
        var_dmp = tuple([variables[i-M] for i in subset] for subset in var_dmp)

        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())

        self.assertEqual(len(var_dmp[0]+var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0]+var_dmp[1]:
            self.assertIn(var, underconstrained_vars)

        self.assertEqual(len(con_dmp[2]), len(underconstrained_cons))
        for con in con_dmp[2]:
            self.assertIn(con, underconstrained_cons)

        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)

        self.assertEqual(len(var_dmp[2]), len(overconstrained_vars))
        for var in var_dmp[2]:
            self.assertIn(var, overconstrained_vars)

        self.assertEqual(len(con_dmp[0]+con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0]+con_dmp[1]:
            self.assertIn(con, overconstrained_cons)

    def test_remove(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))

        igraph = IncidenceGraphInterface(m)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        var_con_set = ComponentSet(igraph.variables + igraph.constraints)
        underconstrained_set = ComponentSet(
            var_dmp.unmatched + var_dmp.underconstrained
        )
        self.assertIn(m.flow_comp[1], var_con_set)
        self.assertIn(m.flow_eqn[1], var_con_set)
        self.assertIn(m.flow_comp[1], underconstrained_set)

        N, M = igraph.incidence_matrix.shape

        # flow_comp[1] is underconstrained, but we think it should be
        # specified by flow_eqn[1], so we remove these from the incidence
        # matrix.
        vars_to_remove = [m.flow_comp[1]]
        cons_to_remove = [m.flow_eqn[1]]
        igraph.remove_nodes(vars_to_remove + cons_to_remove)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        var_con_set = ComponentSet(igraph.variables + igraph.constraints)
        underconstrained_set = ComponentSet(
            var_dmp.unmatched + var_dmp.underconstrained
        )
        self.assertNotIn(m.flow_comp[1], var_con_set)
        self.assertNotIn(m.flow_eqn[1], var_con_set)
        self.assertNotIn(m.flow_comp[1], underconstrained_set)

        N_new, M_new = igraph.incidence_matrix.shape
        self.assertEqual(N_new, N - len(cons_to_remove))
        self.assertEqual(M_new, M - len(vars_to_remove))


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestExtraVars(unittest.TestCase):

    def test_unused_var(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
        igraph = IncidenceGraphInterface(m)
        self.assertEqual(igraph.incidence_matrix.shape, (1, 1))

    def test_reference(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.ref = pyo.Reference(m.v1)
        m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
        igraph = IncidenceGraphInterface(m)
        self.assertEqual(igraph.incidence_matrix.shape, (1, 1))


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
@unittest.skipUnless(AmplInterface.available(), "pynumero_ASL is not available")
class TestExceptions(unittest.TestCase):

    def test_nlp_fixed_error(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.v1 + m.v2 == 1.0)
        m.v2.fix(2.0)
        m._obj = pyo.Objective(expr=0.0)
        nlp = PyomoNLP(m)
        with self.assertRaisesRegex(ValueError, "fixed variables"):
            igraph = IncidenceGraphInterface(nlp, include_fixed=True)

    def test_nlp_active_error(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
        m.c2 = pyo.Constraint(expr=m.v1 == 2.0)
        m._obj = pyo.Objective(expr=0.0)
        nlp = PyomoNLP(m)
        with self.assertRaisesRegex(ValueError, "inactive constraints"):
            igraph = IncidenceGraphInterface(nlp, active=False)

    def test_remove_no_matrix(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        igraph = IncidenceGraphInterface()
        with self.assertRaisesRegex(RuntimeError, "no incidence matrix"):
            igraph.remove_nodes([m.v1])


if __name__ == "__main__":
    unittest.main()
