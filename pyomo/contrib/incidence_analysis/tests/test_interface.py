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
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
    networkx_available,
    plotly_available,
    scipy_available,
)
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
    asl_available,
    IncidenceGraphInterface,
    get_structural_incidence_matrix,
    get_numeric_incidence_matrix,
    get_incidence_graph,
    get_bipartite_incidence_graph,
    extract_bipartite_subgraph,
)
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
    map_coords_to_block_triangular_indices,
)
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
    make_gas_expansion_model,
    make_degenerate_solid_phase_model,
    make_dynamic_model,
)

if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
if networkx_available:
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix

import pyomo.common.unittest as unittest


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
@unittest.skipUnless(asl_available, "pynumero PyomoNLP is not available")
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
        n_var = 4 * (N + 1)
        n_con = 4 * N + 1
        self.assertEqual(imat.shape, (n_con, n_var))

        var_idx_map = ComponentMap((v, i) for i, v in enumerate(all_vars))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(all_cons))

        # Map constraints to the variables they contain.
        csr_map = ComponentMap()
        csr_map.update(
            (
                model.mbal[i],
                ComponentSet(
                    [model.F[i], model.F[i - 1], model.rho[i], model.rho[i - 1]]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (
                model.ebal[i],
                ComponentSet(
                    [
                        model.F[i],
                        model.F[i - 1],
                        model.rho[i],
                        model.rho[i - 1],
                        model.T[i],
                        model.T[i - 1],
                    ]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (
                model.expansion[i],
                ComponentSet(
                    [model.rho[i], model.rho[i - 1], model.P[i], model.P[i - 1]]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (model.ideal_gas[i], ComponentSet([model.P[i], model.rho[i], model.T[i]]))
            for i in model.streams
        )

        # Map constraint and variable indices to the values of the derivatives
        # Note that the derivative values calculated here depend on the model's
        # canonical form.
        deriv_lookup = {}
        m = model  # for convenience
        for s in model.streams:
            # Ideal gas:
            i = con_idx_map[model.ideal_gas[s]]
            j = var_idx_map[model.P[s]]
            deriv_lookup[i, j] = 1.0

            j = var_idx_map[model.rho[s]]
            deriv_lookup[i, j] = -model.R.value * model.T[s].value

            j = var_idx_map[model.T[s]]
            deriv_lookup[i, j] = -model.R.value * model.rho[s].value

            if s != model.streams.first():
                # Expansion:
                i = con_idx_map[model.expansion[s]]
                j = var_idx_map[model.P[s]]
                deriv_lookup[i, j] = 1 / model.P[s - 1].value

                j = var_idx_map[model.P[s - 1]]
                deriv_lookup[i, j] = -model.P[s].value / model.P[s - 1] ** 2

                j = var_idx_map[model.rho[s]]
                deriv_lookup[i, j] = pyo.value(
                    -m.gamma * (m.rho[s] / m.rho[s - 1]) ** (m.gamma - 1) / m.rho[s - 1]
                )

                j = var_idx_map[model.rho[s - 1]]
                deriv_lookup[i, j] = pyo.value(
                    -m.gamma
                    * (m.rho[s] / m.rho[s - 1]) ** (m.gamma - 1)
                    * (-m.rho[s] / m.rho[s - 1] ** 2)
                )

                # Energy balance:
                i = con_idx_map[m.ebal[s]]
                j = var_idx_map[m.rho[s - 1]]
                deriv_lookup[i, j] = pyo.value(m.F[s - 1] * m.T[s - 1])

                j = var_idx_map[m.F[s - 1]]
                deriv_lookup[i, j] = pyo.value(m.rho[s - 1] * m.T[s - 1])

                j = var_idx_map[m.T[s - 1]]
                deriv_lookup[i, j] = pyo.value(m.F[s - 1] * m.rho[s - 1])

                j = var_idx_map[m.rho[s]]
                deriv_lookup[i, j] = pyo.value(-m.F[s] * m.T[s])

                j = var_idx_map[m.F[s]]
                deriv_lookup[i, j] = pyo.value(-m.rho[s] * m.T[s])

                j = var_idx_map[m.T[s]]
                deriv_lookup[i, j] = pyo.value(-m.F[s] * m.rho[s])

                # Mass balance:
                i = con_idx_map[m.mbal[s]]
                j = var_idx_map[m.rho[s - 1]]
                deriv_lookup[i, j] = pyo.value(m.F[s - 1])

                j = var_idx_map[m.F[s - 1]]
                deriv_lookup[i, j] = pyo.value(m.rho[s - 1])

                j = var_idx_map[m.rho[s]]
                deriv_lookup[i, j] = pyo.value(-m.F[s])

                j = var_idx_map[m.F[s]]
                deriv_lookup[i, j] = pyo.value(-m.rho[s])

        # Want to test that the columns have the rows we expect.
        i = model.streams.first()
        for i, j, e in zip(imat.row, imat.col, imat.data):
            con = all_cons[i]
            var = all_vars[j]
            self.assertIn(var, csr_map[con])
            csr_map[con].remove(var)
            self.assertAlmostEqual(pyo.value(deriv_lookup[i, j]), pyo.value(e), 8)
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_numeric_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        n_var = len(variables)
        matching = maximum_matching(imat)
        matching = ComponentMap(
            (c, variables[matching[con_idx_map[c]]]) for c in constraints
        )
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_numeric_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))
        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(imat)
        var_block_map = ComponentMap(
            (v, col_block_map[var_idx_map[v]]) for v in variables
        )
        con_block_map = ComponentMap(
            (c, row_block_map[con_idx_map[c]]) for c in constraints
        )

        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)

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
        n_var = 4 * (N + 1)
        n_con = 4 * N + 1
        self.assertEqual(imat.shape, (n_con, n_var))

        var_idx_map = ComponentMap((v, i) for i, v in enumerate(all_vars))
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(all_cons))

        # Map constraints to the variables they contain.
        csr_map = ComponentMap()
        csr_map.update(
            (
                model.mbal[i],
                ComponentSet(
                    [model.F[i], model.F[i - 1], model.rho[i], model.rho[i - 1]]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (
                model.ebal[i],
                ComponentSet(
                    [
                        model.F[i],
                        model.F[i - 1],
                        model.rho[i],
                        model.rho[i - 1],
                        model.T[i],
                        model.T[i - 1],
                    ]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (
                model.expansion[i],
                ComponentSet(
                    [model.rho[i], model.rho[i - 1], model.P[i], model.P[i - 1]]
                ),
            )
            for i in model.streams
            if i != model.streams.first()
        )
        csr_map.update(
            (model.ideal_gas[i], ComponentSet([model.P[i], model.rho[i], model.T[i]]))
            for i in model.streams
        )

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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))

        n_var = len(variables)
        matching = maximum_matching(imat)
        matching = ComponentMap(
            (c, variables[matching[con_idx_map[c]]]) for c in constraints
        )
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        imat = get_structural_incidence_matrix(variables, constraints)
        con_idx_map = ComponentMap((c, i) for i, c in enumerate(constraints))
        var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))

        row_block_map, col_block_map = map_coords_to_block_triangular_indices(imat)
        var_block_map = ComponentMap(
            (v, col_block_map[var_idx_map[v]]) for v in variables
        )
        con_block_map = ComponentMap(
            (c, row_block_map[con_idx_map[c]]) for c in constraints
        )

        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)

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
@unittest.skipUnless(asl_available, "pynumero PyomoNLP is not available")
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [
            list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)
        ]
        self.assertEqual(len(partition), N + 1)

        for i in model.streams:
            variables = ComponentSet([var for var, _ in partition[i]])
            constraints = ComponentSet([con for _, con in partition[i]])
            if i == model.streams.first():
                self.assertEqual(variables, ComponentSet([model.P[0]]))
            else:
                pred_vars = ComponentSet(
                    [model.rho[i], model.T[i], model.P[i], model.F[i]]
                )
                pred_cons = ComponentSet(
                    [
                        model.ideal_gas[i],
                        model.expansion[i],
                        model.mbal[i],
                        model.ebal[i],
                    ]
                )
                self.assertEqual(pred_vars, variables)
                self.assertEqual(pred_cons, constraints)

    def test_maps_from_triangularization(self):
        N = 5
        model = make_gas_expansion_model(N)
        model.obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(model)
        igraph = IncidenceGraphInterface(nlp)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(
            variables, constraints
        )
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)

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

        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))


@unittest.skipUnless(networkx_available, "networkx is not available.")
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [
            list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)
        ]
        self.assertEqual(len(partition), N + 1)

        for i in model.streams:
            variables = ComponentSet([var for var, _ in partition[i]])
            constraints = ComponentSet([con for _, con in partition[i]])
            if i == model.streams.first():
                self.assertEqual(variables, ComponentSet([model.P[0]]))
            else:
                pred_vars = ComponentSet(
                    [model.rho[i], model.T[i], model.P[i], model.F[i]]
                )
                pred_cons = ComponentSet(
                    [
                        model.ideal_gas[i],
                        model.expansion[i],
                        model.mbal[i],
                        model.ebal[i],
                    ]
                )
                self.assertEqual(pred_vars, variables)
                self.assertEqual(pred_cons, constraints)

    def test_maps_from_triangularization(self):
        """
        This tests the maps from variables and constraints to their diagonal
        blocks returned by map_nodes_to_block_triangular_indices
        """
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(
            variables, constraints
        )
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)

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
        # This test exercises triangularization of a somewhat nontrivial
        # submatrix from a cached incidence matrix.
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of a square,
        # nonsingular subsystem
        variables = []
        half = N // 2
        variables.extend(model.P[i] for i in model.streams if i >= half)
        variables.extend(model.T[i] for i in model.streams if i > half)
        variables.extend(model.rho[i] for i in model.streams if i > half)
        variables.extend(model.F[i] for i in model.streams if i > half)

        constraints = []
        constraints.extend(model.ideal_gas[i] for i in model.streams if i >= half)
        constraints.extend(model.expansion[i] for i in model.streams if i > half)
        constraints.extend(model.mbal[i] for i in model.streams if i > half)
        constraints.extend(model.ebal[i] for i in model.streams if i > half)

        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [
            list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)
        ]
        self.assertEqual(len(partition), (N - half) + 1)

        for i in model.streams:
            idx = i - half
            variables = ComponentSet([var for var, _ in partition[idx]])
            constraints = ComponentSet([con for _, con in partition[idx]])
            if i == half:
                self.assertEqual(variables, ComponentSet([model.P[half]]))
            elif i > half:
                pred_var = ComponentSet(
                    [model.rho[i], model.T[i], model.P[i], model.F[i]]
                )
                pred_con = ComponentSet(
                    [
                        model.ideal_gas[i],
                        model.expansion[i],
                        model.mbal[i],
                        model.ebal[i],
                    ]
                )
                self.assertEqual(variables, pred_var)
                self.assertEqual(constraints, pred_con)

    def test_maps_from_triangularization_submatrix(self):
        # This test exercises the var/con-block-maps obtained from
        # triangularization of a somewhat nontrivial submatrix from a cached
        # incidence matrix.
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)

        # These are the variables and constraints of a square,
        # nonsingular subsystem
        variables = []
        half = N // 2
        variables.extend(model.P[i] for i in model.streams if i >= half)
        variables.extend(model.T[i] for i in model.streams if i > half)
        variables.extend(model.rho[i] for i in model.streams if i > half)
        variables.extend(model.F[i] for i in model.streams if i > half)

        constraints = []
        constraints.extend(model.ideal_gas[i] for i in model.streams if i >= half)
        constraints.extend(model.expansion[i] for i in model.streams if i > half)
        constraints.extend(model.mbal[i] for i in model.streams if i > half)
        constraints.extend(model.ebal[i] for i in model.streams if i > half)

        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(
            variables, constraints
        )
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), (N - half) + 1)
        self.assertEqual(len(con_values), (N - half) + 1)

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

        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

    @unittest.skipUnless(scipy_available, "scipy is not available.")
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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [
            list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)
        ]
        self.assertEqual(len(partition), N + 1)

        for i in model.streams:
            variables = ComponentSet([var for var, _ in partition[i]])
            constraints = ComponentSet([con for _, con in partition[i]])
            if i == model.streams.first():
                self.assertEqual(variables, ComponentSet([model.P[0]]))
            else:
                pred_vars = ComponentSet(
                    [model.rho[i], model.T[i], model.P[i], model.F[i]]
                )
                pred_cons = ComponentSet(
                    [
                        model.ideal_gas[i],
                        model.expansion[i],
                        model.mbal[i],
                        model.ebal[i],
                    ]
                )
                self.assertEqual(pred_vars, variables)
                self.assertEqual(pred_cons, constraints)

    def test_maps_from_triangularization(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(
            variables, constraints
        )
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)

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
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
        # self.assertIs(igraph.row_block_map, None)
        # self.assertIs(igraph.col_block_map, None)
        self.assertEqual(len(var_blocks), N + 1)
        self.assertEqual(len(con_blocks), N + 1)

        for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
            var_set = ComponentSet(vars)
            con_set = ComponentSet(cons)

            if i == 0:
                pred_var_set = ComponentSet([model.P[0]])
                self.assertEqual(pred_var_set, var_set)
                pred_con_set = ComponentSet([model.ideal_gas[0]])
                self.assertEqual(pred_con_set, con_set)

            else:
                pred_var_set = ComponentSet(
                    [model.rho[i], model.T[i], model.P[i], model.F[i]]
                )
                pred_con_set = ComponentSet(
                    [
                        model.ideal_gas[i],
                        model.expansion[i],
                        model.mbal[i],
                        model.ebal[i],
                    ]
                )
                self.assertEqual(pred_var_set, var_set)
                self.assertEqual(pred_con_set, con_set)

    def test_diagonal_blocks_with_cached_maps(self):
        # NOTE: This functionality has been deprecated.
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()

        # These are the variables and constraints of the square,
        # nonsingular subsystem
        variables = []
        variables.extend(model.P.values())
        variables.extend(
            model.T[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.rho[i] for i in model.streams if i != model.streams.first()
        )
        variables.extend(
            model.F[i] for i in model.streams if i != model.streams.first()
        )

        constraints = list(model.component_data_objects(pyo.Constraint))

        igraph.block_triangularize(variables, constraints)
        var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
        # NOTE: row/col_block_map have been deprecated.
        # However, they still return None for now.
        self.assertIs(igraph.row_block_map, None)
        self.assertIs(igraph.col_block_map, None)


@unittest.skipUnless(networkx_available, "networkx is not available.")
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

        self.assertEqual(len(var_dmp[0] + var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0] + var_dmp[1]:
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

        self.assertEqual(len(con_dmp[0] + con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0] + con_dmp[1]:
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

    @unittest.skipUnless(scipy_available, "scipy is not available.")
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
        var_dmp = tuple([variables[i - M] for i in subset] for subset in var_dmp)

        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())

        self.assertEqual(len(var_dmp[0] + var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0] + var_dmp[1]:
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

        self.assertEqual(len(con_dmp[0] + con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0] + con_dmp[1]:
            self.assertIn(con, overconstrained_cons)

    @unittest.skipUnless(scipy_available, "scipy is not available.")
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

    def test_recover_matching_from_dulmage_mendelsohn(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        vdmp, cdmp = igraph.dulmage_mendelsohn()
        vmatch = vdmp.underconstrained + vdmp.square + vdmp.overconstrained
        cmatch = cdmp.underconstrained + cdmp.square + cdmp.overconstrained
        # Assert no duplicates in matched variables and constraints
        self.assertEqual(len(ComponentSet(vmatch)), len(vmatch))
        self.assertEqual(len(ComponentSet(cmatch)), len(cmatch))
        matching = list(zip(vmatch, cmatch))
        # Assert each matched pair contains a variable that participates
        # in the constraint.
        for var, con in matching:
            var_in_con = ComponentSet(igraph.get_adjacent_to(con))
            self.assertIn(var, var_in_con)


@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestConnectedComponents(unittest.TestCase):
    def test_dynamic_model_backward(self):
        """
        This is the same test as performed in the test_connected.py
        file, now implemented with the Pyomo interface.
        """
        m = make_dynamic_model(nfe=5, scheme="BACKWARD")
        m.height[0].fix()
        igraph = IncidenceGraphInterface(m)
        var_blocks, con_blocks = igraph.get_connected_components()
        vc_blocks = [
            (tuple(vars), tuple(cons)) for vars, cons in zip(var_blocks, con_blocks)
        ]
        key_fcn = lambda vc_comps: tuple(
            tuple(comp.name for comp in comps) for comps in vc_comps
        )
        vc_blocks = list(sorted(vc_blocks, key=key_fcn))

        t0_vars = ComponentSet((m.flow_out[0], m.dhdt[0], m.flow_in[0]))
        t0_cons = ComponentSet((m.flow_out_eqn[0], m.diff_eqn[0]))

        # The variables in these blocks need to be sorted by their coordinates
        # in the underlying incidence matrix
        var_key = lambda var: igraph.get_matrix_coord(var)
        con_key = lambda con: igraph.get_matrix_coord(con)
        var_blocks = [
            tuple(sorted(t0_vars, key=var_key)),
            tuple(
                sorted(
                    (var for var in igraph.variables if var not in t0_vars), key=var_key
                )
            ),
        ]
        con_blocks = [
            tuple(sorted(t0_cons, key=con_key)),
            tuple(
                sorted(
                    (con for con in igraph.constraints if con not in t0_cons),
                    key=con_key,
                )
            ),
        ]
        target_blocks = [
            (tuple(vars), tuple(cons)) for vars, cons in zip(var_blocks, con_blocks)
        ]
        target_blocks = list(sorted(target_blocks, key=key_fcn))

        # I am somewhat surprised this works. This appears to because
        # var1 == var2 is a constant equality expression when var1 is var2.
        # So if this test fails, we'll get a somewhat confusing PyomoException
        # about not being able to convert non-constant expressions to bool
        # rather than a message saying that our variables are not the same.
        # self.assertEqual(target_blocks, vc_blocks)
        for block, target_block in zip(vc_blocks, target_blocks):
            vars, cons = block
            pred_vars, pred_cons = target_block
            self.assertEqual(len(vars), len(pred_vars))
            self.assertEqual(len(cons), len(pred_cons))
            for v1, v2 in zip(vars, pred_vars):
                self.assertIs(v1, v2)
            for c1, c2 in zip(cons, pred_cons):
                self.assertIs(c1, c2)


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
class TestExceptions(unittest.TestCase):
    @unittest.skipUnless(scipy_available, "scipy is not available.")
    @unittest.skipUnless(asl_available, "pynumero_ASL is not available")
    def test_nlp_fixed_error(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.v1 + m.v2 == 1.0)
        m.v2.fix(2.0)
        m._obj = pyo.Objective(expr=0.0)
        nlp = PyomoNLP(m)
        msg = "generation options.*are not supported"
        with self.assertRaisesRegex(ValueError, msg):
            igraph = IncidenceGraphInterface(nlp, include_fixed=True)

    @unittest.skipUnless(scipy_available, "scipy is not available.")
    @unittest.skipUnless(asl_available, "pynumero_ASL is not available")
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


@unittest.skipUnless(networkx_available, "networkx is not available.")
@unittest.skipUnless(scipy_available, "scipy is not available.")
class TestIncludeInequality(unittest.TestCase):
    def make_model_with_inequalities(self):
        m = make_degenerate_solid_phase_model()

        @m.Constraint()
        def flow_bound(m):
            return m.flow >= 0

        @m.Constraint(m.components)
        def flow_comp_bound(m, j):
            return m.flow_comp[j] >= 0

        return m

    def test_dont_include_inequality_model(self):
        m = self.make_model_with_inequalities()
        igraph = IncidenceGraphInterface(m, include_inequality=False)
        self.assertEqual(igraph.incidence_matrix.shape, (8, 8))

    def test_include_inequality_model(self):
        m = self.make_model_with_inequalities()
        igraph = IncidenceGraphInterface(m, include_inequality=True)
        self.assertEqual(igraph.incidence_matrix.shape, (12, 8))

    @unittest.skipUnless(asl_available, "pynumero_ASL is not available")
    def test_dont_include_inequality_nlp(self):
        m = self.make_model_with_inequalities()
        m._obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(m)
        igraph = IncidenceGraphInterface(nlp, include_inequality=False)
        self.assertEqual(igraph.incidence_matrix.shape, (8, 8))

    @unittest.skipUnless(asl_available, "pynumero_ASL is not available")
    def test_include_inequality_nlp(self):
        m = self.make_model_with_inequalities()
        m._obj = pyo.Objective(expr=0)
        nlp = PyomoNLP(m)
        igraph = IncidenceGraphInterface(nlp, include_inequality=True)
        self.assertEqual(igraph.incidence_matrix.shape, (12, 8))


@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestGetIncidenceGraph(unittest.TestCase):
    def make_test_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2, 3, 4])
        m.v = pyo.Var(m.I, bounds=(0, None))
        m.eq1 = pyo.Constraint(expr=m.v[1] ** 2 + m.v[2] ** 2 == 1.0)
        m.eq2 = pyo.Constraint(expr=m.v[1] + 2.0 == m.v[3])
        m.ineq1 = pyo.Constraint(expr=m.v[2] - m.v[3] ** 0.5 + m.v[4] ** 2 <= 1.0)
        m.ineq2 = pyo.Constraint(expr=m.v[2] * m.v[4] >= 1.0)
        m.ineq3 = pyo.Constraint(expr=m.v[1] >= m.v[4] ** 4)
        m.obj = pyo.Objective(expr=-m.v[1] - m.v[2] + m.v[3] ** 2 + m.v[4] ** 2)
        return m

    def test_bipartite_incidence_graph(self):
        m = self.make_test_model()
        constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
        variables = list(m.v.values())
        graph = get_bipartite_incidence_graph(variables, constraints)

        # Nodes:
        #   0: m.eq1
        #   1: m.eq2
        #   2: m.ineq1
        #   3: m.ineq2
        #   4: m.ineq3
        #   5: m.v[1]
        #   6: m.v[2]
        #   7: m.v[3]
        #   8: m.v[4]

        # Assert some basic structure
        self.assertEqual(len(graph.nodes), 9)
        self.assertEqual(len(graph.edges), 11)
        self.assertTrue(nx.algorithms.bipartite.is_bipartite(graph))

        # Assert that the "adjacency list" is what we expect
        self.assertEqual(set(graph[0]), {5, 6})
        self.assertEqual(set(graph[1]), {5, 7})
        self.assertEqual(set(graph[2]), {6, 7, 8})
        self.assertEqual(set(graph[3]), {6, 8})
        self.assertEqual(set(graph[4]), {5, 8})
        self.assertEqual(set(graph[5]), {0, 1, 4})
        self.assertEqual(set(graph[6]), {0, 2, 3})
        self.assertEqual(set(graph[7]), {1, 2})
        self.assertEqual(set(graph[8]), {2, 3, 4})

    def test_unused_var(self):
        m = self.make_test_model()
        constraints = [m.eq1, m.eq2]
        variables = list(m.v.values())
        graph = get_bipartite_incidence_graph(variables, constraints)

        # Nodes:
        #   0: m.eq1
        #   1: m.eq2
        #   2: m.v[1]
        #   3: m.v[2]
        #   4: m.v[3]
        #   5: m.v[4]

        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(len(graph.edges), 4)
        self.assertTrue(nx.algorithms.bipartite.is_bipartite(graph))

        # Assert that the "adjacency list" is what we expect
        self.assertEqual(set(graph[0]), {2, 3})
        self.assertEqual(set(graph[1]), {2, 4})
        self.assertEqual(set(graph[2]), {0, 1})
        self.assertEqual(set(graph[3]), {0})
        self.assertEqual(set(graph[4]), {1})
        self.assertEqual(set(graph[5]), set())

    def test_fixed_vars(self):
        m = self.make_test_model()
        constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
        variables = list(m.v.values())
        m.v[1].fix()
        m.v[4].fix()

        # Slightly odd situation where we provide fixed variables, but
        # then tell the graph to not include them. Nodes will be created
        # for these vars, but they will not have any edges.
        graph = get_bipartite_incidence_graph(
            variables, constraints, include_fixed=False
        )

        # Nodes:
        #   0: m.eq1
        #   1: m.eq2
        #   2: m.ineq1
        #   3: m.ineq2
        #   4: m.ineq3
        #   5: m.v[1]
        #   6: m.v[2]
        #   7: m.v[3]
        #   8: m.v[4]

        self.assertEqual(len(graph.nodes), 9)
        self.assertEqual(len(graph.edges), 5)
        self.assertTrue(nx.algorithms.bipartite.is_bipartite(graph))

        # Assert that the "adjacency list" is what we expect
        self.assertEqual(set(graph[0]), {6})
        self.assertEqual(set(graph[1]), {7})
        self.assertEqual(set(graph[2]), {6, 7})
        self.assertEqual(set(graph[3]), {6})
        self.assertEqual(set(graph[4]), set())
        self.assertEqual(set(graph[5]), set())
        self.assertEqual(set(graph[6]), {0, 2, 3})
        self.assertEqual(set(graph[7]), {1, 2})
        self.assertEqual(set(graph[8]), set())

    def test_extract_subgraph(self):
        m = self.make_test_model()
        constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
        variables = list(m.v.values())
        graph = get_bipartite_incidence_graph(variables, constraints)

        sg_cons = [0, 2]
        sg_vars = [i + len(constraints) for i in [2, 0, 3]]

        subgraph = extract_bipartite_subgraph(graph, sg_cons, sg_vars)

        # Subgraph nodes:
        #   0: m.eq1
        #   1: m.ineq1
        #   2: m.v[3]
        #   3: m.v[1]
        #   4: m.v[4]

        self.assertEqual(len(subgraph.nodes), 5)
        self.assertEqual(len(subgraph.edges), 3)
        self.assertTrue(nx.algorithms.bipartite.is_bipartite(subgraph))

        self.assertEqual(set(subgraph[0]), {3})
        self.assertEqual(set(subgraph[1]), {2, 4})
        self.assertEqual(set(subgraph[2]), {1})
        self.assertEqual(set(subgraph[3]), {0})
        self.assertEqual(set(subgraph[4]), {1})

    def test_extract_exceptions(self):
        m = self.make_test_model()
        constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
        variables = list(m.v.values())
        graph = get_bipartite_incidence_graph(variables, constraints)

        sg_cons = [0, 2, 5]
        sg_vars = [i + len(constraints) for i in [2, 3]]
        msg = "Subgraph is not bipartite"
        with self.assertRaisesRegex(RuntimeError, msg):
            subgraph = extract_bipartite_subgraph(graph, sg_cons, sg_vars)

        sg_cons = [0, 2, 5]
        sg_vars = [i + len(constraints) for i in [2, 0, 3]]
        msg = "provided more than once"
        with self.assertRaisesRegex(RuntimeError, msg):
            subgraph = extract_bipartite_subgraph(graph, sg_cons, sg_vars)


@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestGetAdjacent(unittest.TestCase):
    def test_get_adjacent_to_var(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        adj_cons = igraph.get_adjacent_to(m.rho)
        self.assertEqual(
            ComponentSet(adj_cons),
            ComponentSet(
                [m.holdup_eqn[1], m.holdup_eqn[2], m.holdup_eqn[3], m.density_eqn]
            ),
        )

    def test_get_adjacent_to_con(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        adj_vars = igraph.get_adjacent_to(m.density_eqn)
        self.assertEqual(
            ComponentSet(adj_vars), ComponentSet([m.x[1], m.x[2], m.x[3], m.rho])
        )

    def test_get_adjacent_exceptions(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface()
        msg = "Cannot get components adjacent to"
        with self.assertRaisesRegex(RuntimeError, msg):
            adj_vars = igraph.get_adjacent_to(m.density_eqn)

        m.x[1].fix()
        igraph = IncidenceGraphInterface(m, include_fixed=False)
        msg = "Cannot find component"
        with self.assertRaisesRegex(RuntimeError, msg):
            adj_cons = igraph.get_adjacent_to(m.x[1])


@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestInterface(unittest.TestCase):
    def test_assumed_constraint_behavior(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.con = pyo.Constraint(expr=m.x[1] == m.x[2] - pyo.exp(m.x[3]))
        var_set = ComponentSet(identify_variables(m.con.body))
        self.assertEqual(var_set, ComponentSet(m.x[:]))

    def test_subgraph_with_fewer_var_or_con(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=[1, 2])
        m.v = pyo.Var(m.I)
        m.eq1 = pyo.Constraint(expr=m.v[1] + m.v[2] == 1)
        m.ineq1 = pyo.Constraint(expr=m.v[1] - m.v[2] <= 2)

        # Defensively set include_inequality=True, which is the current
        # default, in case this default changes.
        igraph = IncidenceGraphInterface(m, include_inequality=True)

        variables = list(m.v.values())
        constraints = [m.ineq1]
        matching = igraph.maximum_matching(variables, constraints)
        self.assertEqual(len(matching), 1)

        variables = [m.v[2]]
        constraints = [m.eq1, m.ineq1]
        matching = igraph.maximum_matching(variables, constraints)
        self.assertEqual(len(matching), 1)

    @unittest.skipUnless(plotly_available, "Plotly is not available")
    def test_plot(self):
        """
        Unfortunately, this test only ensures the code runs without errors.
        It does not test for correctness.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var()
        m.z = pyo.Var()
        # NOTE: Objective will not be displayed
        m.obj = pyo.Objective(expr=m.y**2 + m.z**2)
        m.c1 = pyo.Constraint(expr=m.y == 2 * m.x + 1)
        m.c2 = pyo.Constraint(expr=m.z >= m.x)
        m.y.fix()
        igraph = IncidenceGraphInterface(m, include_inequality=True, include_fixed=True)
        igraph.plot(title='test plot', show=False)

    def test_zero_coeff(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.eq1 = pyo.Constraint(expr=m.x[1] + 0 * m.x[2] == 2)
        m.eq2 = pyo.Constraint(expr=m.x[1] ** 2 == 1)
        m.eq3 = pyo.Constraint(expr=m.x[2] * m.x[3] - m.x[1] == 1)

        igraph = IncidenceGraphInterface(m)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()

        # Because 0*m.x[2] does not appear in the incidence graph, we correctly
        # identify that the system is structurally singular
        self.assertGreater(len(var_dmp.unmatched), 0)

    def test_var_minus_itself(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.eq1 = pyo.Constraint(expr=m.x[1] + m.x[2] - m.x[2] == 2)
        m.eq2 = pyo.Constraint(expr=m.x[1] ** 2 == 1)
        m.eq3 = pyo.Constraint(expr=m.x[2] * m.x[3] - m.x[1] == 1)

        igraph = IncidenceGraphInterface(m)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()

        # m.x[2] - m.x[2] is correctly ignored by generate_standard_repn,
        # so we correctly identify that the system is structurally singular
        self.assertGreater(len(var_dmp.unmatched), 0)

    def test_linear_only(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.eq1 = pyo.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 == 1)
        m.eq2 = pyo.Constraint(expr=m.x[2] + pyo.sqrt(m.x[1]) + pyo.exp(m.x[3]) == 1)
        m.eq3 = pyo.Constraint(expr=m.x[3] + m.x[1] ** 3 + m.x[2] == 1)

        igraph = IncidenceGraphInterface(m, linear_only=True)
        self.assertEqual(igraph.n_edges, 3)
        self.assertEqual(ComponentSet(igraph.variables), ComponentSet([m.x[2], m.x[3]]))

        matching = igraph.maximum_matching()
        self.assertEqual(len(matching), 2)
        self.assertIs(matching[m.eq2], m.x[2])
        self.assertIs(matching[m.eq3], m.x[3])


@unittest.skipUnless(networkx_available, "networkx is not available.")
class TestIndexedBlock(unittest.TestCase):
    def test_block_data_obj(self):
        m = pyo.ConcreteModel()
        m.block = pyo.Block([1, 2, 3])
        m.block[1].subblock = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m.block[1])
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        self.assertEqual(len(var_dmp.unmatched), 1)
        self.assertEqual(len(con_dmp.unmatched), 1)

        msg = "Unsupported type.*_BlockData"
        with self.assertRaisesRegex(TypeError, msg):
            igraph = IncidenceGraphInterface(m.block)


if __name__ == "__main__":
    unittest.main()
