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
        )
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.asl import AmplInterface

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

if __name__ == "__main__":
    unittest.main()
